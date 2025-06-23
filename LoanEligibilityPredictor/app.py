from loan_model import LoanEligibilityPredictor
import streamlit as st
import pandas as pd

# Initialize predictor
predictor = LoanEligibilityPredictor()
predictor.load_saved_model()

# Streamlit UI
st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")

# Custom CSS - Comprehensive text visibility fix
st.markdown("""
    <style>
    /* Set ALL text to black by default */
    html, body, [class*="css"]  {
        color: #000000;
    }
    
    /* Specific element enhancements */
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white !important;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .result {
        font-size: 20px;
        font-weight: bold;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .approved {
        background-color: #d4edda;
        color: #155724 !important;
    }
    .rejected {
        background-color: #f8d7da;
        color: #721c24 !important;
    }
    
    /* Make all headers and labels clearly visible */
    h1, h2, h3, h4, h5, h6, .stMarkdown, .stSubheader, .stTitle {
        color: #000000 !important;
    }
    
    /* Form labels and inputs */
    .stForm label, .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #000000 !important;
        font-weight: bold;
        font-size: 1rem;
    }
    
    /* Select box and input text */
    .stSelectbox, .stNumberInput, .stTextInput {
        color: #000000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# App title
st.title("Loan Eligibility Predictor")

# Input form
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information", divider='gray')
        gender = st.selectbox("Gender", ["Male", "Female"], key='gender')
        married = st.selectbox("Marital Status", ["Yes", "No"], key='married')
        dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3], key='dependents')
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], key='education')
        self_employed = st.selectbox("Self Employed", ["Yes", "No"], key='self_employed')
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], key='property_area')
        
    with col2:
        st.subheader("Financial Information", divider='gray')
        applicant_income = st.number_input("Applicant Income (USD)", min_value=0, value=5000, key='applicant_income')
        coapplicant_income = st.number_input("Coapplicant Income (USD)", min_value=0, value=2000, key='coapplicant_income')
        loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=150, key='loan_amount')
        loan_term = st.number_input("Loan Term (days)", min_value=0, value=360, key='loan_term')
        credit_history = st.selectbox("Credit History meets guidelines", [1, 0], key='credit_history')
    
    submitted = st.form_submit_button("Predict Loan Eligibility")

# Prediction logic
if submitted:
    input_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }
    
    try:
        prediction, probability = predictor.predict_loan_eligibility(input_data)
        
        if prediction == 'Y':
            st.markdown(f"""
            <div class="result approved">
                Loan Approved with {probability*100:.1f}% confidence!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result rejected">
                Loan Rejected with {(1-probability)*100:.1f}% confidence.
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")