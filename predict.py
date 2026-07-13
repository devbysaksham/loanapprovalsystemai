import joblib
import numpy as np
import os

def load_artifacts():
    """
    Loads the saved Machine Learning model and the encoders.
    """
    model_path = 'models/loan_model.pkl'
    encoder_path = 'models/encoder.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError("Model or Encoder artifacts not found. Please run train_model.py first.")
        
    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    
    return model, encoders

def predict_loan(input_dict):
    """
    Accepts user input as a dictionary, processes it using the saved encoders,
    and returns the prediction and probability.
    
    input_dict should contain:
    - Gender
    - Married
    - Dependents
    - Education
    - Self_Employed
    - ApplicantIncome
    - CoapplicantIncome
    - LoanAmount
    - Loan_Amount_Term
    - Credit_History
    - Property_Area
    """
    model, encoders = load_artifacts()
    
    # Process categorical variables using the loaded encoders
    processed_input = []
    
    # Define exact feature order as expected by the model
    features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                'Loan_Amount_Term', 'Credit_History', 'Property_Area']
                
    for feature in features:
        val = input_dict[feature]
        if feature in encoders:
            # We must handle 'Dependents' special case since "3+" might be passed as string
            val_encoded = encoders[feature].transform([str(val)])[0]
            processed_input.append(val_encoded)
        else:
            processed_input.append(float(val))
            
    input_array = np.array([processed_input])
    
    # Make prediction
    prediction_num = model.predict(input_array)[0]
    probability_arr = model.predict_proba(input_array)[0]
    
    # Decode the output
    # By default, LabelEncoder sorts alphabetically: N=0, Y=1
    status = "Approved" if prediction_num == 1 else "Rejected"
    
    confidence = probability_arr[1] if prediction_num == 1 else probability_arr[0]
    
    return status, confidence
