import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Bank Term Deposit Predictor", page_icon="", layout="centered")

# Load the saved model assets (Model, Preprocessor, Features, Threshold)
@st.cache_resource
def load_assets():
    try:
        return joblib.load('bank_model_assets.pkl')
    except FileNotFoundError:
        st.error("'bank_model_assets.pkl' file not found.")
        return None

assets = load_assets()

if assets:
    model = assets['model']
    preprocessor = assets['preprocessor']
    selected_features = assets['selected_features']
    threshold = assets['threshold']
else:
    st.stop()


# 2. USER INTERFACE (SIDEBAR & MAIN FORM)

st.title("🏦 Bank Telemarketing Prediction App")
st.markdown(f"""
This application predicts whether a client will subscribe to a term deposit.
**Optimal Decision Threshold:** `{threshold:.2f}` (Adjusted for high recall)
""")

st.sidebar.header("📝 Client Information")

def user_input_features():

    st.sidebar.subheader("Demographics")
    age = st.sidebar.number_input("Age", min_value=17, max_value=98, value=30)
    job = st.sidebar.selectbox("Job", 
        ['admin.', 'blue-collar', 'technician', 'services', 'management', 
         'retired', 'entrepreneur', 'self-employed', 'housemaid', 'unemployed', 'student', 'unknown'])
    marital = st.sidebar.selectbox("Marital Status", ['married', 'single', 'divorced', 'unknown'])
    education = st.sidebar.selectbox("Education", 
        ['university.degree', 'high.school', 'basic.9y', 'professional.course', 
         'basic.4y', 'basic.6y', 'unknown', 'illiterate'])
    
    # --- Financial Status ---
    st.sidebar.subheader("Financial Status")
    default = st.sidebar.selectbox("Credit Default?", ['no', 'unknown', 'yes'])
    housing = st.sidebar.selectbox("Housing Loan?", ['yes', 'no', 'unknown'])
    loan = st.sidebar.selectbox("Personal Loan?", ['no', 'yes', 'unknown'])

    # --- Campaign History ---
    st.sidebar.subheader("Campaign History")
    contact = st.sidebar.selectbox("Contact Type", ['cellular', 'telephone'])
    month = st.sidebar.selectbox("Last Contact Month", 
        ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep', 'mar', 'dec'])
    day_of_week = st.sidebar.selectbox("Last Contact Day", ['thu', 'mon', 'wed', 'tue', 'fri'])
    campaign = st.sidebar.number_input("Number of Contacts (Current Campaign)", min_value=1, value=1)
    pdays = st.sidebar.number_input("Days passed since last campaign (999 means never contacted)", value=999)
    previous = st.sidebar.number_input("Number of Contacts (Previous Campaigns)", value=0)
    poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", ['nonexistent', 'failure', 'success'])

    # --- Socio-Economic Indicators ---
    st.sidebar.subheader("Economic Indicators")
    emp_var_rate = st.sidebar.number_input("Employment Variation Rate", value=-1.8)
    cons_price_idx = st.sidebar.number_input("Consumer Price Index", value=92.8)
    cons_conf_idx = st.sidebar.number_input("Consumer Confidence Index", value=-46.2)
    euribor3m = st.sidebar.number_input("Euribor 3 Month Rate", value=1.2)
    nr_employed = st.sidebar.number_input("Number of Employees", value=5099.1)

    # Combine into a DataFrame
    data = {
        'age': age, 'job': job, 'marital': marital, 'education': education,
        'default': default, 'housing': housing, 'loan': loan,
        'contact': contact, 'month': month, 'day_of_week': day_of_week,
        'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        'emp.var.rate': emp_var_rate, 'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx, 'euribor3m': euribor3m, 'nr.employed': nr_employed
    }
    return pd.DataFrame([data])

# Get input from user
input_df = user_input_features()

# Show input Summary
st.subheader("Client Data Summary")
st.write(input_df)


if st.button("Predict Subscription"):
    
    # A. Preprocessing 
    processed_array = preprocessor.transform(input_df)
    
    # B. Restore Feature Names
    # The preprocessor output is a numpy array. We need to convert it back to a DataFrame
    # with the correct column names so we can select the specific 15 features.
    feature_names = preprocessor.get_feature_names_out()
    processed_df = pd.DataFrame(processed_array, columns=feature_names)
    
    # C. Feature Selection
    # Select only the 15 features used during training
    try:
        final_input = processed_df[selected_features]
    except KeyError as e:
        st.error(f"The model expects specific features that were selected during training. Error: {e}")
        st.stop()

    # D. Prediction
    prediction_proba = model.predict_proba(final_input)[:, 1][0]
    
    # Apply our Custom Threshold
    prediction = 1 if prediction_proba >= threshold else 0


    st.markdown("---")
    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(label="Subscription Probability", value=f"{prediction_proba:.2%}")

    with col2:
        if prediction == 1:
            st.success("**YES, POTENTIAL SUBSCRIBER!**")
            st.markdown(f"The probability is above the threshold of **{threshold:.2f}**.")
        else:
            st.error("**NO, UNLIKELY TO SUBSCRIBE**")
            st.markdown(f"The probability is below the threshold of **{threshold:.2f}**.")


    st.progress(float(prediction_proba))