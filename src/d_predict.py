import joblib
import pandas as pd
from utils import preprocess_new

# Load artifacts
model = joblib.load('E:\Projects\Credit Card Fraud detection\models\model.pkl')
encoder = joblib.load('E:\Projects\Credit Card Fraud detection\models\encoder.pkl')

def predict_fraud(transaction):
    """Predict fraud probability for a transaction"""
    try:
        processed = preprocess_new(transaction, encoder)
        proba = model.predict_proba(processed)[0][1]
        return {
            'fraud_probability': round(proba, 4),
            'is_fraud': int(proba > 0.6),
            'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

if __name__ == "__main__":
    # Test prediction
    sample_tx = {
        'amount': 1000.00,
        'merchant': 'TechHaven',
        'category': 'Groceries',
        'location': 'BR',
        'device': 'Mobile',
        'time': '2025-05-29T03:45:21'
    }
    print("Sample Fraud Prediction:")
    print(predict_fraud(sample_tx))