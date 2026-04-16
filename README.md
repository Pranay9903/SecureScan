# Zero-Day Phishing Detection System

Advanced phishing detection using NLP and Machine Learning with ensemble stacking (BERT, LSTM, XGBoost, Random Forest).

## Features
- Real-time URL phishing detection
- 98%+ accuracy with ensemble models
- User authentication & history tracking
- Interactive charts & visualizations
- REST API for developers
- Admin panel for management
- Login required popup for non-authenticated users

## Quick Start (Local)

```bash
pip install -r requirements.txt
python -m app.app
```

Open http://localhost:5000

**Default Admin:** `admin` / `admin123`

## Deployment

### Render
1. Push to GitHub
2. Create Web Service on Render
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn app.app:app --workers 4 --bind 0.0.0.0:$PORT`

### PythonAnywhere
1. Clone repo
2. Create virtual environment & install requirements
3. Configure WSGI file
4. Reload

## Tech Stack
- Python 3.x, Flask, SQLAlchemy
- PyTorch, Transformers (BERT)
- Scikit-learn, XGBoost
- Chart.js for visualizations

## License
MIT
