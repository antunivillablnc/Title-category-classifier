# Academic Paper Classifier

A Flask-based web application that classifies academic papers into different subject areas using TF-IDF and Logistic Regression.

## Features

- Classifies academic papers based on title and abstract
- Uses TF-IDF vectorization and Logistic Regression
- Provides confidence scores for predictions
- Modern, responsive web interface
- Easy to deploy and extend

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Deployment

### Local Deployment
The application can be run locally using Flask's development server:
```bash
python app.py
```

### Production Deployment
For production deployment, it's recommended to use Gunicorn:

1. Install Gunicorn (already included in requirements.txt)
2. Run the application:
```bash
gunicorn app:app
```

### Cloud Deployment
The application can be deployed to various cloud platforms:

#### Heroku
1. Create a `Procfile`:
```
web: gunicorn app:app
```

2. Deploy to Heroku:
```bash
heroku create
git push heroku main
```

#### AWS Elastic Beanstalk
1. Create a `requirements.txt` file (already done)
2. Create a `.ebextensions/python.config` file:
```yaml
option_settings:
  aws:elasticbeanstalk:container:python:
    WSGIPath: app:app
```

3. Deploy using the AWS EB CLI:
```bash
eb init
eb create
eb deploy
```

## Model Training

The current implementation includes a simple training dataset. To improve the model:

1. Add more training data to the `SAMPLE_DATA` dictionary in `app.py`
2. Adjust the `TfidfVectorizer` parameters
3. Modify the `LogisticRegression` parameters

## API Usage

The classification endpoint accepts POST requests to `/predict` with the following JSON structure:

```json
{
    "title": "Paper Title",
    "abstract": "Paper Abstract"
}
```

Response format:
```json
{
    "success": true,
    "predictions": [
        {
            "category": "Category Name",
            "probability": 0.95
        }
    ]
}
```

## Contributing

Feel free to submit issues and enhancement requests! 