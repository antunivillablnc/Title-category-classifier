from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, session
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
from functools import wraps
from datetime import datetime, timedelta
from collections import Counter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Required for sessions

# Initialize the model and vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
model = LogisticRegression(max_iter=1000)

# Sample categories for demonstration
CATEGORIES = [
    'Machine Learning',
    'Statistics',
    'Finance',
    'Computer Science',
    'Mathematics',
    'Physics',
    'Biology',
    'Chemistry'
]

# Sample training data (in a real application, this would be loaded from a database)
SAMPLE_DATA = {
    'Machine Learning': [
        'Deep Learning Approaches for Image Recognition',
        'Neural Networks in Natural Language Processing',
        'Reinforcement Learning for Game Playing'
    ],
    'Statistics': [
        'Bayesian Methods in Data Analysis',
        'Statistical Inference for Large Datasets',
        'Time Series Analysis and Forecasting'
    ],
    'Finance': [
        'Quantitative Trading Strategies',
        'Risk Management in Financial Markets',
        'Portfolio Optimization Techniques'
    ]
}

# In-memory history log
history_log = []

# Simple basic auth
USERNAME = 'professor'
PASSWORD = 'secret123'

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"',
         'Cache-Control': 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def train_model():
    # Prepare training data
    texts = []
    labels = []
    for category, papers in SAMPLE_DATA.items():
        texts.extend(papers)
        labels.extend([category] * len(papers))
    
    # Fit vectorizer and transform texts
    X = vectorizer.fit_transform(texts)
    
    # Train model
    model.fit(X, labels)
    
    # Save the model and vectorizer
    joblib.dump(model, 'model.joblib')
    joblib.dump(vectorizer, 'vectorizer.joblib')

# Train the model if it doesn't exist
if not os.path.exists('model.joblib'):
    train_model()
else:
    model = joblib.load('model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')

@app.route('/')
def home():
    return render_template('index.html', categories=CATEGORIES)

def get_predictions(title, abstract):
    text = f"{title} {abstract}"
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]
    return [
        {
            'category': model.classes_[idx],
            'probability': float(probabilities[idx])
        }
        for idx in top_indices
    ]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        
        # Check for duplicates
        is_duplicate = any(
            entry['title'] == title and entry['abstract'] == abstract 
            for entry in history_log
        )
        
        if not is_duplicate:
            # Get predictions
            predictions = get_predictions(title, abstract)
            
            # Add to history log
            history_log.append({
                'title': title,
                'abstract': abstract,
                'predictions': predictions,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return jsonify({
                'success': True,
                'predictions': predictions
            })
        else:
            # If duplicate, just return predictions without adding to history
            predictions = get_predictions(title, abstract)
            return jsonify({
                'success': True,
                'predictions': predictions,
                'message': 'This paper was already classified before'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/history')
@requires_auth
def history():
    # Calculate statistics
    total_papers = len(history_log)
    
    # Get unique categories
    all_categories = []
    for entry in history_log:
        all_categories.extend([pred['category'] for pred in entry['predictions']])
    unique_categories = list(set(all_categories))
    
    # Calculate average confidence
    total_confidence = 0
    confidence_count = 0
    for entry in history_log:
        for pred in entry['predictions']:
            total_confidence += pred['probability'] * 100
            confidence_count += 1
    avg_confidence = total_confidence / confidence_count if confidence_count > 0 else 0
    
    # Count recent papers (last 24 hours)
    recent_time = datetime.now() - timedelta(hours=24)
    recent_count = sum(1 for entry in history_log 
                      if datetime.strptime(entry['timestamp'], '%Y-%m-%d %H:%M:%S') > recent_time)
    
    # Get category distribution
    category_counter = Counter(all_categories)
    top_categories = category_counter.most_common(5)
    
    # Prepare data for pie chart
    category_labels = list(category_counter.keys())
    category_counts = list(category_counter.values())
    
    return render_template('history.html',
                         history=history_log,
                         unique_categories=unique_categories,
                         avg_confidence=avg_confidence,
                         recent_count=recent_count,
                         top_categories=top_categories,
                         category_labels=category_labels,
                         category_counts=category_counts)

if __name__ == '__main__':
    app.run(debug=True) 