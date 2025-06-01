import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

def preprocess_data():
    df = pd.read_csv('E:\Projects\Credit Card Fraud detection\data\generated_data.csv')
    
    # Feature engineering
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour
    df['day_of_week'] = df['time'].dt.dayofweek
    
    # Encode categoricals
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = encoder.fit_transform(df[['category', 'location', 'device']])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
    
    # Final dataset
    features = pd.concat([df[['amount', 'hour', 'day_of_week']], encoded_df], axis=1)
    target = df['is_fraud']
    
    # Handle imbalance
    X_res, y_res = SMOTE().fit_resample(features, target)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )
    
    # Save artifacts
    joblib.dump(encoder, 'E:\Projects\Credit Card Fraud detection\models\encoder.pkl')
    pd.concat([X_test, y_test], axis=1).to_csv('E:\Projects\Credit Card Fraud detection\data\processed_data.csv', index=False)
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_data()
    print("Data preprocessing complete!")