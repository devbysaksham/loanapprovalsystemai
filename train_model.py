import pandas as pd
import numpy as np
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

def main():
    print("Starting Model Training Pipeline...")
    
    # 1. Load Dataset
    dataset_path = 'datasets/train_u6lujuX_CVtuZ9i.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    df = pd.read_csv(dataset_path)
    print(f"Dataset loaded successfully with shape: {df.shape}")

    # 2. Preprocess Dataset (Handling missing values)
    print("Preprocessing Dataset...")
    
    # Fill categorical missing values with Mode
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Loan_Amount_Term', 'Credit_History']
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
        
    # Fill numerical missing values with Median
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    # Drop Loan_ID as it's not useful for prediction
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    # 3. Encode Categorical Variables
    print("Encoding Categorical Variables...")
    
    cat_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    encoders = {}
    
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Save the encoder mapping logic
    os.makedirs('models', exist_ok=True)
    joblib.dump(encoders, 'models/encoder.pkl')
    print("Encoders saved to models/encoder.pkl")

    # 4. Split Data
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")

    # 5. Train Best Model (Random Forest selected for robust performance)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Evaluate Model
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\n========================================")
    print("EVALUATION METRICS")
    print("========================================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Model
    joblib.dump(model, 'models/loan_model.pkl')
    print("Model successfully saved to models/loan_model.pkl")
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    main()
