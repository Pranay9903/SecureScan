import os
import sys
import re
from urllib.parse import urlparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import secrets
import json

from src.pipeline import PhishingDetectionPipeline
from src.features import URLFeatureExtractor
from app.models import db, User, ScanHistory, Feedback, ApiKey, init_db

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(32)
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'phishing_detection.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ECHO'] = False

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.session_protection = 'strong'

pipeline = None
url_extractor = URLFeatureExtractor()

def validate_url(url):
    if not url or len(url) > 2048:
        return False, "URL is too long or empty"
    
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return False, "Invalid URL format"
        
        if parsed.scheme not in ('http', 'https'):
            return False, "Only HTTP and HTTPS are allowed"
        
        localhost_patterns = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
        if any(pattern in parsed.netloc.lower() for pattern in localhost_patterns):
            return False, "Localhost URLs are not allowed"
        
        return True, url
    except:
        return False, "Invalid URL"

def check_password_strength(password):
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain letters"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain numbers"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain special characters"
    return True, "Password is strong"


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def init_app():
    global pipeline
    init_db(app)
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'phishing_model.joblib')
    
    if os.path.exists(model_path):
        try:
            pipeline = PhishingDetectionPipeline()
            pipeline.load(model_path)
        except Exception as e:
            print(f"Could not load model: {e}")


def fetch_html_selenium(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
    
    driver = None
    html_content = None
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        driver.get(url)
        
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        time.sleep(2)
        html_content = driver.page_source
        
    except Exception as e:
        print(f"Selenium error: {e}")
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            html_content = response.text
        except:
            html_content = None
    finally:
        if driver:
            driver.quit()
    
    return html_content


def fetch_html_requests(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10, verify=False)
        return response.text
    except:
        return None


def analyze_url(url, html_content=None):
    features = url_extractor.extract(url)
    
    if pipeline and pipeline.is_trained:
        try:
            result = pipeline.predict(url, html_content)
            features['model_prediction'] = result['prediction']
            features['model_confidence'] = result['confidence']
            features['individual_predictions'] = result.get('individual_predictions', {})
        except Exception as e:
            features['model_prediction'] = 'Unknown'
            features['model_confidence'] = 0.0
            features['individual_predictions'] = {}
    else:
        score = 0
        if features.get('has_ip', 0) == 1:
            score += 30
        if features.get('suspicious_tld', 0) == 1:
            score += 25
        if features.get('urgency_score', 0) > 0:
            score += min(features['urgency_score'] * 10, 20)
        if features.get('num_at', 0) > 0:
            score += 20
        if features.get('num_underscores', 0) > 3:
            score += 15
        
        phishing_prob = min(score / 100, 1.0)
        
        features['model_prediction'] = 'Phishing' if phishing_prob > 0.5 else 'Safe'
        features['model_confidence'] = abs(phishing_prob - 0.5) * 2 + 0.5
        features['individual_predictions'] = {}
    
    return features


@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
@login_required
def detect():
    data = request.get_json()
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'Please enter a URL'}), 400
    
    is_valid, result = validate_url(url)
    if not is_valid:
        return jsonify({'error': result}), 400
    
    url = result
    
    try:
        html_content = fetch_html_selenium(url)
        if html_content is None:
            html_content = fetch_html_requests(url)
        
        features = analyze_url(url, html_content)
        
        if current_user.is_authenticated:
            scan = ScanHistory(
                user_id=current_user.id,
                url=url,
                prediction=features['model_prediction'],
                confidence=features['model_confidence'],
                url_length=features.get('url_length'),
                domain_length=features.get('domain_length'),
                has_https=features.get('has_https', 0) == 1,
                has_ip=features.get('has_ip', 0) == 1,
                num_dots=features.get('num_dots'),
                num_hyphens=features.get('num_hyphens'),
                num_underscores=features.get('num_underscores'),
                num_at=features.get('num_at'),
                num_params=features.get('num_params'),
                num_subdomains=features.get('num_subdomains'),
                urgency_score=features.get('urgency_score')
            )
            
            if features.get('individual_predictions'):
                ind = features['individual_predictions']
                scan.rf_probability = ind.get('rf', {}).get('probability')
                scan.xgb_probability = ind.get('xgb', {}).get('probability')
                scan.lstm_probability = ind.get('lstm', {}).get('probability')
                scan.bert_probability = ind.get('bert', {}).get('probability')
                scan.ensemble_probability = features['model_confidence']
            
            db.session.add(scan)
            db.session.commit()
            scan_saved = True
        else:
            scan_saved = False
        
        return jsonify({
            'prediction': features['model_prediction'],
            'confidence': features['model_confidence'],
            'scan_saved': scan_saved,
            'url_features': {
                'url_length': features.get('url_length'),
                'domain_length': features.get('domain_length'),
                'has_https': features.get('has_https'),
                'has_ip': features.get('has_ip'),
                'num_dots': features.get('num_dots'),
                'num_hyphens': features.get('num_hyphens'),
                'num_subdomains': features.get('num_subdomains'),
                'urgency_score': features.get('urgency_score'),
                'suspicious_tld': features.get('suspicious_tld')
            },
            'individual_predictions': features.get('individual_predictions', {})
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing URL: {str(e)}'}), 500


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    data = request.get_json()
    url = data.get('url', '').strip()
    scan_type = data.get('scan_type', 'quick')
    
    if not url:
        return jsonify({'error': 'Please enter a URL'}), 400
    
    is_valid, result = validate_url(url)
    if not is_valid:
        return jsonify({'error': result}), 400
    
    url = result
    
    try:
        html_content = None
        if scan_type == 'detailed':
            html_content = fetch_html_selenium(url)
            if html_content is None:
                html_content = fetch_html_requests(url)
        
        features = analyze_url(url, html_content)
        
        scan = ScanHistory(
            user_id=current_user.id,
            url=url,
            prediction=features['model_prediction'],
            confidence=features['model_confidence'],
            scan_type=scan_type,
            url_length=features.get('url_length'),
            domain_length=features.get('domain_length'),
            has_https=features.get('has_https', 0) == 1,
            has_ip=features.get('has_ip', 0) == 1,
            num_dots=features.get('num_dots'),
            num_hyphens=features.get('num_hyphens'),
            num_underscores=features.get('num_underscores'),
            num_at=features.get('num_at'),
            num_params=features.get('num_params'),
            num_subdomains=features.get('num_subdomains'),
            urgency_score=features.get('urgency_score')
        )
        
        if features.get('individual_predictions'):
            ind = features['individual_predictions']
            scan.rf_probability = ind.get('rf', {}).get('probability', 0.5)
            scan.xgb_probability = ind.get('xgb', {}).get('probability', 0.5)
            scan.lstm_probability = ind.get('lstm', {}).get('probability', 0.5)
            scan.bert_probability = ind.get('bert', {}).get('probability', 0.5)
            scan.ensemble_probability = features['model_confidence']
        
        db.session.add(scan)
        db.session.commit()
        
        return jsonify({
            'id': scan.id,
            'prediction': features['model_prediction'],
            'confidence': features['model_confidence'],
            'scan_saved': True,
            'url_features': features,
            'individual_predictions': features.get('individual_predictions', {}),
            'created_at': scan.created_at.isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        data = request.form
        
        if len(data['username']) < 3:
            flash('Username must be at least 3 characters', 'error')
            return redirect(url_for('register'))
        
        if len(data['username']) > 50:
            flash('Username is too long', 'error')
            return redirect(url_for('register'))
        
        if not re.match(r'^[a-zA-Z0-9_]+$', data['username']):
            flash('Username can only contain letters, numbers, and underscores', 'error')
            return redirect(url_for('register'))
        
        is_valid_password, password_msg = check_password_strength(data['password'])
        if not is_valid_password:
            flash(password_msg, 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(username=data['username']).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=data['email']).first():
            flash('Email already registered', 'error')
            return redirect(url_for('register'))
        
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', data['email']):
            flash('Invalid email format', 'error')
            return redirect(url_for('register'))
        
        if data['password'] != data['confirm_password']:
            flash('Passwords do not match', 'error')
            return redirect(url_for('register'))
        
        user = User(
            username=data['username'],
            email=data['email']
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.form
        user = User.query.filter_by(username=data['username']).first()
        
        if user and user.check_password(data['password']):
            if not user.is_active:
                flash('Account is deactivated', 'error')
                return redirect(url_for('login'))
            
            remember_me = data.get('remember') == 'on'
            login_user(user, remember=remember_me)
            user.last_login = db.func.now()
            db.session.commit()
            
            if user.is_admin:
                return redirect(url_for('admin_dashboard'))
            
            next_page = request.args.get('next')
            if next_page:
                return redirect(next_page)
            return redirect(url_for('dashboard'))
        
        flash('Invalid username or password', 'error')
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/dashboard')
@login_required
def dashboard():
    recent_scans = ScanHistory.query.filter_by(user_id=current_user.id).order_by(ScanHistory.created_at.desc()).limit(10).all()
    
    total_scans = ScanHistory.query.filter_by(user_id=current_user.id).count()
    phishing_count = ScanHistory.query.filter_by(user_id=current_user.id, prediction='Phishing').count()
    safe_count = ScanHistory.query.filter_by(user_id=current_user.id, prediction='Safe').count()
    
    stats = {
        'total': total_scans,
        'phishing': phishing_count,
        'safe': safe_count,
        'accuracy': round((phishing_count / total_scans * 100) if total_scans > 0 else 0, 1)
    }
    
    return render_template('dashboard.html', scans=recent_scans, stats=stats)


@app.route('/history')
@login_required
def history():
    scans = ScanHistory.query.filter_by(user_id=current_user.id).order_by(ScanHistory.created_at.desc()).all()
    return render_template('history.html', scans=scans)


@app.route('/scan/<int:scan_id>')
@login_required
def scan_detail(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and not current_user.is_admin:
        flash('Access denied', 'error')
        return redirect(url_for('history'))
    
    return render_template('scan_detail.html', scan=scan)


@app.route('/feedback', methods=['POST'])
@login_required
def feedback():
    data = request.form
    scan = ScanHistory.query.get(data['scan_id'])
    
    if not scan or (scan.user_id != current_user.id and not current_user.is_admin):
        flash('Invalid scan', 'error')
        return redirect(url_for('history'))
    
    fb = Feedback(
        user_id=current_user.id,
        scan_id=scan.id,
        is_correct=data.get('is_correct') == 'true',
        comment=data.get('comment', '')
    )
    
    db.session.add(fb)
    db.session.commit()
    
    flash('Thank you for your feedback!', 'success')
    return redirect(url_for('scan_detail', scan_id=scan.id))


@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    total_users = User.query.count()
    total_scans = ScanHistory.query.count()
    phishing_scans = ScanHistory.query.filter_by(prediction='Phishing').count()
    safe_scans = ScanHistory.query.filter_by(prediction='Safe').count()
    
    recent_scans = ScanHistory.query.order_by(ScanHistory.created_at.desc()).limit(20).all()
    users = User.query.order_by(User.created_at.desc()).limit(10).all()
    
    stats = {
        'total_users': total_users,
        'total_scans': total_scans,
        'phishing_scans': phishing_scans,
        'safe_scans': safe_scans
    }
    
    return render_template('admin_dashboard.html', stats=stats, scans=recent_scans, users=users)


@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin_users.html', users=users)


@app.route('/admin/user/<int:user_id>/toggle', methods=['POST'])
@login_required
def toggle_user(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    user = User.query.get_or_404(user_id)
    user.is_active = not user.is_active
    db.session.commit()
    
    return jsonify({'success': True, 'is_active': user.is_active})


@app.route('/admin/scans')
@login_required
def admin_scans():
    if not current_user.is_admin:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    scans = ScanHistory.query.order_by(ScanHistory.created_at.desc()).limit(100).all()
    return render_template('admin_scans.html', scans=scans)


@app.route('/admin/feedback')
@login_required
def admin_feedback():
    if not current_user.is_admin:
        flash('Access denied', 'error')
        return redirect(url_for('dashboard'))
    
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).limit(50).all()
    return render_template('admin_feedback.html', feedbacks=feedbacks)


@app.route('/api-keys')
@login_required
def api_keys():
    keys = ApiKey.query.filter_by(user_id=current_user.id).all()
    return render_template('api_keys.html', keys=keys)


@app.route('/api-keys/create', methods=['POST'])
@login_required
def create_api_key():
    key = secrets.token_hex(32)
    api_key = ApiKey(
        user_id=current_user.id,
        key=key,
        name=request.form.get('name', 'Default Key')
    )
    db.session.add(api_key)
    db.session.commit()
    
    flash(f'API Key created: {key}', 'success')
    return redirect(url_for('api_keys'))


@app.route('/api-keys/<int:key_id>/revoke', methods=['POST'])
@login_required
def revoke_api_key(key_id):
    api_key = ApiKey.query.get_or_404(key_id)
    if api_key.user_id != current_user.id and not current_user.is_admin:
        flash('Access denied', 'error')
        return redirect(url_for('api_keys'))
    
    api_key.is_active = False
    db.session.commit()
    
    flash('API Key revoked', 'success')
    return redirect(url_for('api_keys'))


@app.route('/api/v1/detect', methods=['POST'])
def api_detect():
    api_key = request.headers.get('X-API-Key')
    
    if api_key:
        key_obj = ApiKey.query.filter_by(key=api_key, is_active=True).first()
        if key_obj:
            key_obj.usage_count += 1
            key_obj.last_login = db.func.now()
            db.session.commit()
    
    data = request.get_json()
    url = data.get('url', '').strip()
    
    if not url:
        return jsonify({'error': 'Please provide a URL'}), 400
    
    is_valid, result = validate_url(url)
    if not is_valid:
        return jsonify({'error': result}), 400
    
    url = result
    
    try:
        html_content = fetch_html_requests(url)
        features = analyze_url(url, html_content)
        
        return jsonify({
            'url': url,
            'prediction': features['model_prediction'],
            'confidence': features['model_confidence'],
            'features': {
                'url_length': features.get('url_length'),
                'has_https': features.get('has_https'),
                'has_ip': features.get('has_ip'),
                'suspicious_tld': features.get('suspicious_tld')
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/stats')
@login_required
def api_stats():
    if not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    stats = {
        'total_users': User.query.count(),
        'total_scans': ScanHistory.query.count(),
        'phishing_detected': ScanHistory.query.filter_by(prediction='Phishing').count(),
        'safe_detected': ScanHistory.query.filter_by(prediction='Safe').count()
    }
    
    return jsonify(stats)


@app.route('/export/history')
@login_required
def export_history():
    scans = ScanHistory.query.filter_by(user_id=current_user.id).all()
    
    csv_content = "ID,URL,Prediction,Confidence,Created At\n"
    for scan in scans:
        csv_content += f"{scan.id},{scan.url},{scan.prediction},{scan.confidence},{scan.created_at}\n"
    
    return csv_content, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': 'attachment; filename=scan_history.csv'
    }


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/features')
def features_page():
    return render_template('features.html')


@app.route('/batch-scan', methods=['POST'])
@login_required
def batch_scan():
    data = request.get_json()
    urls = data.get('urls', [])
    
    if not urls or len(urls) > 50:
        return jsonify({'error': 'Provide 1-50 URLs for batch scanning'}), 400
    
    results = []
    for url in urls:
        url = url.strip()
        if not url:
            continue
            
        is_valid, result = validate_url(url)
        if not is_valid:
            results.append({'url': url, 'error': result, 'prediction': 'Invalid', 'confidence': 0})
            continue
        
        url = result
        try:
            html_content = fetch_html_requests(url)
            features = analyze_url(url, html_content)
            results.append({
                'url': url,
                'prediction': features['model_prediction'],
                'confidence': features['model_confidence'],
                'is_suspicious': features.get('suspicious_tld', 0) == 1 or features.get('has_ip', 0) == 1
            })
        except Exception as e:
            results.append({'url': url, 'error': str(e), 'prediction': 'Error', 'confidence': 0})
    
    for r in results:
        if 'prediction' in r and r['prediction'] != 'Invalid' and r['prediction'] != 'Error':
            scan = ScanHistory(
                user_id=current_user.id,
                url=r['url'],
                prediction=r['prediction'],
                confidence=r['confidence'],
                scan_type='batch'
            )
            db.session.add(scan)
    
    db.session.commit()
    
    return jsonify({
        'total': len(results),
        'scanned': len([r for r in results if 'prediction' in r and r['prediction'] not in ['Invalid', 'Error']]),
        'results': results
    })


@app.route('/export/json/<int:scan_id>')
@login_required
def export_scan_json(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    report = {
        'id': scan.id,
        'url': scan.url,
        'prediction': scan.prediction,
        'confidence': scan.confidence,
        'scan_type': scan.scan_type,
        'url_features': {
            'url_length': scan.url_length,
            'domain_length': scan.domain_length,
            'has_https': scan.has_https,
            'has_ip': scan.has_ip,
            'num_dots': scan.num_dots,
            'num_hyphens': scan.num_hyphens,
            'num_underscores': scan.num_underscores,
            'num_at': scan.num_at,
            'num_params': scan.num_params,
            'num_subdomains': scan.num_subdomains,
            'urgency_score': scan.urgency_score
        },
        'model_predictions': {
            'rf': scan.rf_probability,
            'xgb': scan.xgb_probability,
            'lstm': scan.lstm_probability,
            'bert': scan.bert_probability,
            'ensemble': scan.ensemble_probability
        },
        'created_at': scan.created_at.isoformat() if scan.created_at else None
    }
    
    return jsonify(report)


@app.route('/export/pdf/<int:scan_id>')
@login_required
def export_scan_pdf(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id and not current_user.is_admin:
        return jsonify({'error': 'Access denied'}), 403
    
    html_content = f"""
    <html>
    <head><title>Scan Report - {scan.id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 40px; }}
        h1 {{ color: #6366f1; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8fafc; border-radius: 8px; }}
        .label {{ font-weight: bold; color: #64748b; }}
        .value {{ color: #1e293b; }}
        .safe {{ color: #10b981; }}
        .phishing {{ color: #ef4444; }}
    </style>
    </head>
    <body>
        <h1>SecureScan Report #{scan.id}</h1>
        <div class="section">
            <p><span class="label">URL:</span> <span class="value">{scan.url}</span></p>
            <p><span class="label">Prediction:</span> <span class="{scan.prediction.lower()}">{scan.prediction}</span></p>
            <p><span class="label">Confidence:</span> <span class="value">{(scan.confidence * 100):.1f}%</span></p>
            <p><span class="label">Date:</span> <span class="value">{scan.created_at.strftime('%Y-%m-%d %H:%M')}</span></p>
        </div>
        <div class="section">
            <h3>URL Features</h3>
            <p><span class="label">URL Length:</span> <span class="value">{scan.url_length}</span></p>
            <p><span class="label">Has HTTPS:</span> <span class="value">{'Yes' if scan.has_https else 'No'}</span></p>
            <p><span class="label">Contains IP:</span> <span class="value">{'Yes' if scan.has_ip else 'No'}</span></p>
            <p><span class="label">Subdomains:</span> <span class="value">{scan.num_subdomains}</span></p>
        </div>
        <p style="color: #64748b; font-size: 12px;">Generated by SecureScan Phishing Detection System</p>
    </body>
    </html>
    """
    
    return html_content, 200, {'Content-Type': 'text/html'}


@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    if request.method == 'POST':
        data = request.form
        
        if 'new_password' in data and data['new_password']:
            is_valid, msg = check_password_strength(data['new_password'])
            if not is_valid:
                flash(msg, 'error')
                return redirect(url_for('profile'))
            
            if 'confirm_password' not in data or data['new_password'] != data['confirm_password']:
                flash('Passwords do not match', 'error')
                return redirect(url_for('profile'))
            
            current_user.set_password(data['new_password'])
        
        if 'username' in data and data['username'] != current_user.username:
            if User.query.filter_by(username=data['username']).first():
                flash('Username already taken', 'error')
                return redirect(url_for('profile'))
            current_user.username = data['username']
        
        if 'email' in data and data['email'] != current_user.email:
            if User.query.filter_by(email=data['email']).first():
                flash('Email already in use', 'error')
                return redirect(url_for('profile'))
            current_user.email = data['email']
        
        db.session.commit()
        flash('Profile updated successfully', 'success')
        return redirect(url_for('profile'))
    
    scans_count = ScanHistory.query.filter_by(user_id=current_user.id).count()
    return render_template('profile.html', scans_count=scans_count)


@app.route('/account/delete', methods=['POST'])
@login_required
def delete_account():
    user_id = current_user.id
    logout_user()
    
    user = User.query.get(user_id)
    if user:
        db.session.delete(user)
        db.session.commit()
    
    return redirect(url_for('index'))


@app.route('/bookmarks')
@login_required
def bookmarks():
    from app.models import Bookmark
    bookmarks_list = Bookmark.query.filter_by(user_id=current_user.id).order_by(Bookmark.created_at.desc()).all()
    return render_template('bookmarks.html', bookmarks=bookmarks_list)


@app.route('/bookmarks/<int:scan_id>/add', methods=['POST'])
@login_required
def add_bookmark(scan_id):
    scan = ScanHistory.query.get_or_404(scan_id)
    if scan.user_id != current_user.id:
        return jsonify({'error': 'Access denied'}), 403
    
    from app.models import Bookmark
    existing = Bookmark.query.filter_by(user_id=current_user.id, scan_id=scan_id).first()
    if existing:
        return jsonify({'error': 'Already bookmarked'}), 400
    
    bookmark = Bookmark(user_id=current_user.id, scan_id=scan_id)
    db.session.add(bookmark)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Added to bookmarks'})


@app.route('/bookmarks/<int:scan_id>/remove', methods=['POST'])
@login_required
def remove_bookmark(scan_id):
    from app.models import Bookmark
    bookmark = Bookmark.query.filter_by(user_id=current_user.id, scan_id=scan_id).first()
    if bookmark:
        db.session.delete(bookmark)
        db.session.commit()
    
    return jsonify({'success': True, 'message': 'Removed from bookmarks'})


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    init_app()
    app.run(debug=True, port=5000)