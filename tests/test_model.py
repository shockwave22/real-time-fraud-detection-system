import joblib
import numpy as np

def test_model_quality():
    model = joblib.load('../models/model.pkl')
    # Mock feature array matching training shape
    mock_data = np.zeros((1, 20))  
    mock_data[0, 0] = 1000  # Amount feature
    
    pred = model.predict_proba(mock_data)
    assert 0 <= pred[0][0] <= 1, "Invalid prediction"
    print("✅ Model validation passed!")

if __name__ == "__main__":
    test_model_quality()