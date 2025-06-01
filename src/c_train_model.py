from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
from utils import preprocess_new
import pandas as pd
from b_preprocess import preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = preprocess_data()
    
    model = XGBClassifier(
        scale_pos_weight=10,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        eval_metric='aucpr',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluation
    print("\nModel Evaluation:")
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))
    print(f"AUC-ROC: {roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")
    
    joblib.dump(model, 'E:\Projects\Credit Card Fraud detection\models\model.pkl')
    print("Model saved to models/model.pkl")

if __name__ == "__main__":
    train_model()