import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Load model
# -------------------------
model = joblib.load('rf_credit_risk_model.pkl')

# -------------------------
# Title
# -------------------------
st.title('💳 Credit Risk Prediction App')
st.write('Enter customer information:')

st.sidebar.title("Customer Input")

# -------------------------
# User-friendly mappings
# -------------------------

active_member_map = {
    "Yes": 1,
    "No": 0
}

digital_behavior_map = {
    "Offline": 1,
    "Mobile": 0
}

customer_segment_map = {
    "Mass": 0,
    "Emerging": 1,
    "Priority": 2,
    "Affluent": 3
}

# -------------------------
# Input widgets with description
# -------------------------

last_transaction_month = st.number_input(
    "Last Month Transaction Amount (VND) — Total amount of money transacted last month",
    min_value=0,
    value=0,
    step=1000000
)

days_since_last_active = st.number_input(
    "Days Since Last Active",
    min_value=0,
    value=30
)

active_member_label = st.selectbox(
    "Active Member",
    ["Yes", "No"]
)

digital_behavior_label = st.selectbox(
    "Digital Behavior",
    ["Mobile", "Offline"]
)

nums_service = st.number_input(
    "Number of Services Used",
    min_value=0,
    value=2
)

balance = st.number_input(
    "Current account Balance (VND)",
    min_value=0,
    value=5000000,
    step=1000000
)

customer_segment_label = st.selectbox(
    "Customer tier",
    ["Mass", "Emerging", "Priority", "Affluent"]
)

monthly_ir = st.number_input(
    "Monthly Income (VND)",
    min_value=0,
    value=10000000,
    step=1000000
)

# -------------------------
# Convert labels to encoded values
# -------------------------

active_member = active_member_map[active_member_label]
digital_behavior = digital_behavior_map[digital_behavior_label]
customer_segment = customer_segment_map[customer_segment_label]

# -------------------------
# Create dataframe
# -------------------------

input_dict = {
    'last_transaction_month': last_transaction_month,
    'days_since_last_active': days_since_last_active,
    'active_member': active_member,
    'digital_behavior': digital_behavior,
    'nums_service': nums_service,
    'balance': balance,
    'customer_segment': customer_segment,
    'monthly_ir': monthly_ir
}

input_df = pd.DataFrame([input_dict])

# Show input preview
st.write("Input Data:")
st.dataframe(input_df)

# -------------------------
# Prediction button
# -------------------------

if st.button('Predict Risk'):

    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.write(f'Risk Probability: {prob:.2%}')

    # Progress bar
    st.progress(float(prob))

    # Risk level interpretation
    if prob > 0.7:
        st.error("🔴 Very High Risk")
    elif prob > 0.4:
        st.warning("🟡 Moderate Risk")
    else:
        st.success("🟢 Low Risk")

    # Binary prediction display
    if pred == 1:
        st.error('Model Prediction: High Credit Risk')
    else:
        st.success('Model Prediction: Low Credit Risk')

        