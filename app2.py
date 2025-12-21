import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# CONFIGURATION & SETUP
# ==========================================
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide"
)

# Load the saved model package
# Uses st.cache_resource to load the model only once and speed up the app
@st.cache_resource
def load_model():
    try:
        model_package = joblib.load('bank_model_pipeline.pkl')
        return model_package
    except FileNotFoundError:
        st.error("Error: 'bank_model_pipeline.pkl' not found. Please run the notebook first to generate the model.")
        return None

package = load_model()

# ==========================================
# UI DESIGN - HEADER
# ==========================================
st.title("🏦 Bank Marketing Prediction App")
st.markdown("""
This application predicts whether a client will subscribe to a term deposit. 
Adjust the customer details below and click **Predict**.
""")

if package:
    pipeline = package['pipeline']
    threshold = package['threshold']
    
    # ==========================================
    # INPUT FORM
    # ==========================================
    st.sidebar.header("Customer Information")
    
    with st.form("prediction_form"):
        st.subheader("1. Client Profile")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            job = st.selectbox("Job", [
                'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                'retired', 'self-employed', 'services', 'student', 'technician', 
                'unemployed', 'unknown'
            ])
            marital = st.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
        
        with col2:
            education = st.selectbox("Education", [
                'basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 
                'professional.course', 'university.degree', 'unknown'
            ])
            default = st.selectbox("Has Credit in Default?", ['no', 'yes', 'unknown'])
            housing = st.selectbox("Has Housing Loan?", ['no', 'yes', 'unknown'])
        
        with col3:
            loan = st.selectbox("Has Personal Loan?", ['no', 'yes', 'unknown'])

        st.divider()
        st.subheader("2. Contact & Campaign Info")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            contact = st.selectbox("Contact Type", ['cellular', 'telephone'])
            month = st.selectbox("Month", [
                'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
            ])
            day_of_week = st.selectbox("Day of Week", ['mon', 'tue', 'wed', 'thu', 'fri'])
            
        with col5:
            campaign = st.number_input("Number of Contacts (Current Campaign)", min_value=1, value=1)
            pdays = st.number_input("Days since last contact (999 means never)", min_value=0, value=999)
            previous = st.number_input("Number of Contacts (Before this campaign)", min_value=0, value=0)
            
        with col6:
            poutcome = st.selectbox("Outcome of Previous Campaign", ['nonexistent', 'failure', 'success'])

        st.divider()
        st.subheader("3. Socio-Economic Indicators")
        col7, col8, col9 = st.columns(3)
        
        with col7:
            emp_var_rate = st.number_input("Employment Variation Rate", value=-1.8)
            cons_price_idx = st.number_input("Consumer Price Index", value=92.893)
        
        with col8:
            cons_conf_idx = st.number_input("Consumer Confidence Index", value=-46.2)
            euribor3m = st.number_input("Euribor 3 Month Rate", value=1.313)
            
        with col9:
            nr_employed = st.number_input("Number of Employees", value=5099.1)

        # Submit Button
        submitted = st.form_submit_button("🚀 Predict Subscription")

    # ==========================================
    # PREDICTION LOGIC
    # ==========================================
    if submitted:
        # 1. Create a DataFrame from inputs
        input_data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'month': [month],
            'day_of_week': [day_of_week],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [nr_employed]
        })

        # 2. FEATURE ENGINEERING (Must match Notebook Cell 34)
        # We must replicate the exact feature creation logic used in training
        input_data['contacted_before'] = input_data['pdays'].apply(lambda x: 0 if x == 999 else 1)
        input_data['interaction_score'] = input_data['campaign'] + input_data['previous']

        # 3. Predict using the Pipeline
        try:
            # We use predict_proba + Custom Threshold
            probs = pipeline.predict_proba(input_data)[:, 1]
            prediction = (probs >= threshold).astype(int)[0]
            probability = probs[0]

            # 4. Display Results
            st.divider()
            if prediction == 1:
                st.success("## ✅ Result: SUBSCRIBED")
                st.write(f"The model predicts this client is **likely** to subscribe.")
                st.metric(label="Probability of Subscription", value=f"{probability:.2%}")
            else:
                st.error("## ❌ Result: NOT SUBSCRIBED")
                st.write(f"The model predicts this client is **unlikely** to subscribe.")
                st.metric(label="Probability of Subscription", value=f"{probability:.2%}")
                
            st.caption(f"Note: Classification Threshold used: {threshold:.4f}")
            
            # Show input data for verification
            with st.expander("See Raw Input Data"):
                st.dataframe(input_data)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
else:
    st.stop()