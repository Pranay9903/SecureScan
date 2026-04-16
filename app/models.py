from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    scans = db.relationship('ScanHistory', backref='user', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'is_admin': self.is_admin,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ScanHistory(db.Model):
    __tablename__ = 'scan_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    url = db.Column(db.String(2048), nullable=False)
    prediction = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    scan_type = db.Column(db.String(20), default='quick')
    
    rf_probability = db.Column(db.Float)
    xgb_probability = db.Column(db.Float)
    lstm_probability = db.Column(db.Float)
    bert_probability = db.Column(db.Float)
    ensemble_probability = db.Column(db.Float)
    
    url_length = db.Column(db.Integer)
    domain_length = db.Column(db.Integer)
    has_https = db.Column(db.Boolean)
    has_ip = db.Column(db.Boolean)
    num_dots = db.Column(db.Integer)
    num_hyphens = db.Column(db.Integer)
    num_underscores = db.Column(db.Integer)
    num_at = db.Column(db.Integer)
    num_params = db.Column(db.Integer)
    num_subdomains = db.Column(db.Integer)
    urgency_score = db.Column(db.Integer)
    
    html_features = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'url': self.url,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'scan_type': self.scan_type,
            'individual_predictions': {
                'rf': self.rf_probability,
                'xgb': self.xgb_probability,
                'lstm': self.lstm_probability,
                'bert': self.bert_probability,
                'ensemble': self.ensemble_probability
            },
            'url_features': {
                'url_length': self.url_length,
                'domain_length': self.domain_length,
                'has_https': self.has_https,
                'has_ip': self.has_ip,
                'num_dots': self.num_dots,
                'num_hyphens': self.num_hyphens,
                'num_underscores': self.num_underscores,
                'num_at': self.num_at,
                'num_params': self.num_params,
                'num_subdomains': self.num_subdomains,
                'urgency_score': self.urgency_score
            },
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class UrlReport(db.Model):
    __tablename__ = 'url_reports'
    
    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scan_history.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    
    report_type = db.Column(db.String(20), default='detailed')
    report_data = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Feedback(db.Model):
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    scan_id = db.Column(db.Integer, db.ForeignKey('scan_history.id'), nullable=True)
    
    is_correct = db.Column(db.Boolean, nullable=True)
    comment = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class ApiKey(db.Model):
    __tablename__ = 'api_keys'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    key = db.Column(db.String(64), unique=True, nullable=False)
    name = db.Column(db.String(100))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used = db.Column(db.DateTime)
    usage_count = db.Column(db.Integer, default=0)
    
    user = db.relationship('User', backref='api_keys')


def init_db(app):
    with app.app_context():
        db.create_all()
        
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(username='admin', email='admin@phishingdetector.com', is_admin=True)
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created (admin/admin123)")