import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="MBA Admission Predictor", layout="centered")

st.title("MBA Admission Predictor")
st.markdown("**Enter applicant details below** — get an instant Admit/Deny prediction with probability.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.5, step=0.01)
    gmat = st.number_input("GMAT Score", min_value=200, max_value=800, value=650, step=10)
    work_exp = st.number_input("Work Experience (years)", min_value=0, max_value=30, value=5, step=1)

with col2:
    international = st.selectbox("International Student?", ["No", "Yes"])
    race = st.selectbox("Race/Ethnicity", ["Unknown", "Black", "White", "Asian", "Hispanic", "Other"])
    major = st.selectbox("Major", ["Business", "STEM", "Humanities"])   
    work_industry = st.selectbox("Work Industry", ['Financial Services', 'Investment Management', 'Technology', 'Consulting', 'Nonprofit/Gov', 'PE/VC', 'Health Care', 'Investment Banking', 'Other',
 'Retail', 'Energy', 'CPG', 'Real Estate', 'Media/Entertainment']) 

# ——— PREDICT BUTTON ———
if st.button("Predict Admission", type="primary"):
    input_dict = {
        'gender': 1 if gender == "Male" else 0,
        'international': 1 if international == "Yes" else 0,
        'gpa': gpa,
        'major': major,
        'race': race,
        'gmat': gmat,
        'work_exp': work_exp,
        'work_industry': work_industry
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Apply exact same preprocessing as notebook
    input_df = pd.get_dummies(input_df, drop_first=True)
    
    # Load training column order and align (this is the pro trick)
    feature_columns = joblib.load('feature_columns.pkl')
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Load artifacts
    model = joblib.load('log_reg_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    input_scaled = scaler.transform(input_df)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.success(f"**ADMIT** ✅\n\nProbability of admission: **{probability:.1%}**")
    else:
        st.error(f"**DENY** ❌\n\nProbability of admission: **{probability:.1%}**")
    
    st.caption("Model metrics from your notebook were preserved (Logistic Regression + StandardScaler).")