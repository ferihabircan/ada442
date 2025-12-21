import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Bank Deposit AI",
    page_icon="💸",
    layout="centered", # 'centered' looks better for forms than 'wide'
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CONSTANTS (For Clean Code)
# ==========================================
JOBS = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
        'retired', 'self-employed', 'services', 'student', 'technician', 
        'unemployed', 'unknown']
MARITAL = ['married', 'single', 'divorced', 'unknown']
EDUCATION = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 
             'professional.course', 'university.degree', 'unknown']
MONTHS = ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
DAYS = ['mon', 'tue', 'wed', 'thu', 'fri']
POUTCOMES = ['nonexistent', 'failure', 'success']

# ==========================================
# 3. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model_pipeline():
    filename = 'bank_model_pipeline.pkl'
    try:
        return joblib.load(filename)
    except FileNotFoundError:
        return None

package = load_model_pipeline()

# ==========================================
# 4. SIDEBAR & HEADER
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
st.sidebar.title("App Guide")
st.sidebar.info(
    """
    **How to use:**
    1. Fill in the client details in the tabs.
    2. Check financial indicators.
    3. Click **Analyze Client**.
    
    **Model:** Random Forest w/ Pipeline
    """
)

st.title("💸 AI Bank Marketing Predictor")
st.markdown("### Will the customer subscribe to a term deposit?")
st.markdown("Use this tool to analyze customer potential based on historical data.")

if package is None:
    st.error("⚠️ **Model file not found!** Please run **Cell 51** in your Jupyter Notebook to generate `bank_model_pipeline.pkl`.")
    st.stop()

pipeline = package['pipeline']
threshold = package['threshold']

# ==========================================
# 5. MAIN FORM (Tabbed Layout)
# ==========================================
with st.form("prediction_form"):
    
    # Create Tabs for better organization
    tab1, tab2, tab3 = st.tabs(["👤 Client Profile", "📉 Economics", "📞 Campaign History"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 18, 95, 30, help="Client's age in years")
            job = st.selectbox("Job Type", JOBS)
            marital = st.selectbox("Marital Status", MARITAL)
        
        with col2:
            education = st.selectbox("Education Level", EDUCATION)
            housing = st.radio("Has Housing Loan?", ['no', 'yes', 'unknown'], horizontal=True)
            loan = st.radio("Has Personal Loan?", ['no', 'yes', 'unknown'], horizontal=True)
            default = st.selectbox("Credit in Default?", ['no', 'yes', 'unknown'])

    with tab2:
        st.caption("Macroeconomic indicators at the time of contact")
        col3, col4 = st.columns(2)
        with col3:
            emp_var_rate = st.slider("Employment Var. Rate", -4.0, 2.0, -1.8, 0.1)
            cons_price_idx = st.number_input("Consumer Price Index", 90.0, 95.0, 92.893, step=0.001)
            cons_conf_idx = st.slider("Consumer Confidence", -55.0, -20.0, -46.2, 0.1)
        with col4:
            euribor3m = st.number_input("Euribor 3 Month Rate", 0.0, 6.0, 1.313, step=0.001)
            nr_employed = st.number_input("No. of Employees", 4900.0, 5300.0, 5099.1, step=10.0)

    with tab3:
        col5, col6 = st.columns(2)
        with col5:
            contact = st.selectbox("Contact Method", ['cellular', 'telephone'])
            month = st.selectbox("Last Contact Month", MONTHS, index=2) # Default May
            day_of_week = st.selectbox("Day of Week", DAYS)
        
        with col6:
            campaign = st.slider("Contacts (This Campaign)", 1, 50, 1)
            previous = st.slider("Contacts (Before this)", 0, 10, 0)
            pdays = st.number_input("Days since previous contact", 0, 999, 999, help="999 means never contacted")
            poutcome = st.selectbox("Previous Outcome", POUTCOMES)

    # Centered Submit Button
    submit_btn = st.form_submit_button("✨ Analyze Client Potential", type="primary")

# ==========================================
# 6. PREDICTION LOGIC & RESULTS
# ==========================================
if submit_btn:
    # A. Data Preparation
    input_df = pd.DataFrame({
        'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
        'default': [default], 'housing': [housing], 'loan': [loan],
        'contact': [contact], 'month': [month], 'day_of_week': [day_of_week],
        'campaign': [campaign], 'pdays': [pdays], 'previous': [previous],
        'poutcome': [poutcome], 'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx], 'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m], 'nr.employed': [nr_employed]
    })

    # B. Feature Engineering (Critical Step!)
    input_df['contacted_before'] = input_df['pdays'].apply(lambda x: 0 if x == 999 else 1)
    input_df['interaction_score'] = input_df['campaign'] + input_df['previous']

    # C. Prediction
    try:
        # Get probability (0 to 1)
        probs = pipeline.predict_proba(input_df)[:, 1]
        probability = probs[0]
        
        # Apply Optimized Threshold
        prediction = 1 if probability >= threshold else 0

        # D. Display
        st.divider()
        st.subheader("Prediction Results")

        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if prediction == 1:
                st.balloons() # FUN FACTOR
                st.success("## ✅ YES")
                st.write("Client is likely to subscribe.")
            else:
                st.error("## ❌ NO")
                st.write("Client is unlikely to subscribe.")

        with col_res2:
            st.write(f"**Confidence Score:** {probability:.2%}")
            
            # Custom Progress Bar color logic
            if probability > threshold:
                bar_color = "green"
            else:
                bar_color = "red"
                
            st.progress(probability, text="Subscription Probability")
            
            if probability > threshold:
                st.info("💡 **Tip:** This client matches the profile of previous subscribers. Prepare the contract!")
            else:
                st.warning("💡 **Tip:** Focus on other clients or try a different campaign strategy.")

    except Exception as e:
        st.error(f"Error during calculation: {e}")