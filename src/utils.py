import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder

def preprocess_new(transaction, encoder):
    """Preprocess new transaction for prediction"""
    df = pd.DataFrame([transaction])
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
    
    # Encode categoricals
    encoded = encoder.transform(df[['category', 'location', 'device']])
    encoded_df = pd.DataFrame(encoded, 
                             columns=encoder.get_feature_names_out())
    
    return pd.concat([df[['amount', 'hour', 'day_of_week']], encoded_df], axis=1)