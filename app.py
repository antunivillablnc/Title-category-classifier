from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        
        # Combine title and abstract
        text = f"{title} {abstract}"
        
        # Transform the input text
        X = vectorizer.transform([text])
        
        # Make prediction
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        # Get top 3 predictions with probabilities
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [
            {
                'category': model.classes_[idx],
                'probability': float(probabilities[idx])
            }
            for idx in top_indices
        ]
        
        return jsonify({
            'success': True,
            'predictions': top_predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 