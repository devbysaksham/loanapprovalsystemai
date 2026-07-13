import streamlit as st
from predict import predict_loan

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism, Theme, and Animations
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Gradient Header */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 40px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.5);
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        margin-bottom: 20px;
    }

    /* Result Card - Approved */
    .result-approved {
        background: rgba(40, 167, 69, 0.15);
        backdrop-filter: blur(10px);
        border: 2px solid #28a745;
        border-left: 10px solid #28a745;
        border-radius: 15px;
        padding: 25px;
        color: #155724;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(40, 167, 69, 0.2);
        animation: fadeIn 0.5s ease-in-out;
    }

    /* Result Card - Rejected */
    .result-rejected {
        background: rgba(220, 53, 69, 0.15);
        backdrop-filter: blur(10px);
        border: 2px solid #dc3545;
        border-left: 10px solid #dc3545;
        border-radius: 15px;
        padding: 25px;
        color: #721c24;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(220, 53, 69, 0.2);
        animation: fadeIn 0.5s ease-in-out;
    }

    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Nice Typography */
    h1, h2, h3 {
        font-weight: 600;
        color: #1e3c72;
    }
    .main-header h1 {
        color: white;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        color: #666;
        font-size: 0.9rem;
        margin-top: 50px;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Project Info", "Developer Info"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Information")
st.sidebar.info(
    "Algorithm: Random Forest\n\n"
    "Target: Loan_Status\n\n"
    "Est. Accuracy: ~80%+"
)

# ----------------- MAIN APP -----------------
if page == "Home":
    st.markdown('<div class="main-header"><h1>🏦 AI-Based Loan Approval System</h1><p style="font-size:1.2rem; margin-top:10px;">Predict whether a customer\'s loan application should be Approved or Rejected using Machine Learning.</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card"><h3>Applicant Details</h3>', unsafe_allow_html=True)
    
    # Form using columns
    with st.form("loan_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            
        with col2:
            self_employed = st.selectbox("Self Employed", ["No", "Yes"])
            applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=500)
            coapplicant_income = st.number_input("Coapplicant Income ($)", min_value=0, value=0, step=500)
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            
        with col3:
            loan_amount = st.number_input("Loan Amount (in Thousands)", min_value=1, value=150, step=10)
            loan_amount_term = st.number_input("Loan Amount Term (Months)", min_value=12, value=360, step=12)
            credit_history = st.selectbox("Credit History", ["1.0 (Good)", "0.0 (Bad)"])
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        submit_col, reset_col = st.columns([1, 1])
        with submit_col:
            submit = st.form_submit_button("Predict Loan Approval", use_container_width=True)
        with reset_col:
            reset = st.form_submit_button("Reset Form", use_container_width=True)
            
    st.markdown('</div>', unsafe_allow_html=True) # End glass card

    if reset:
        st.rerun()

    if submit:
        # Prepare dictionary for predict module
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': dependents,
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': 1.0 if "1.0" in credit_history else 0.0,
            'Property_Area': property_area
        }
        
        try:
            status, confidence = predict_loan(input_data)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if status == "Approved":
                st.balloons()
                st.markdown(f"""
                <div class="result-approved">
                    <h1 style="color: #155724;">✅ Loan Approved</h1>
                    <h3>Confidence Score: {confidence * 100:.2f}%</h3>
                    <p>Congratulations! Based on the model's analysis, this loan application is highly likely to be approved.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-rejected">
                    <h1 style="color: #721c24;">❌ Loan Rejected</h1>
                    <h3>Confidence Score: {confidence * 100:.2f}%</h3>
                    <p>We're sorry, but based on the provided information, the model suggests this loan application will be rejected.</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

elif page == "Project Info":
    st.title("📊 Project Information")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("""
    ### AI-Based Loan Approval System
    This project is a Machine Learning application that leverages financial and demographic data to predict loan approval.
    
    #### Features:
    - **Machine Learning**: Uses Scikit-learn Random Forest for high accuracy predictions.
    - **Data Pipeline**: Handles missing values, categorical encoding, and serialization using Joblib.
    - **Web App**: Built entirely in Python using Streamlit with Custom CSS, Glassmorphism, and responsive layouts.
    
    #### Technology Stack:
    - Python
    - Pandas, NumPy
    - Scikit-learn
    - Streamlit
    """)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Developer Info":
    st.title("👨‍💻 Developer Information")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("""
    ### About the Developer
    This project was built as a comprehensive real-world internship assignment demonstrating end-to-end Machine Learning lifecycle, from Data Analysis and Model Training to Web Deployment.
    
    * Designed for production readiness.
    * GitHub compatible.
    * Fully Deployable on Streamlit Community Cloud.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">AI-Based Loan Approval System © 2026 | Internship Project Submission</div>', unsafe_allow_html=True)
