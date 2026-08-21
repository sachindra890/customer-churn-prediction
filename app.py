import streamlit as st
import joblib
import pandas as pd


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Retention Intelligence",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #9ca3af;
    margin-bottom: 30px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 18px;
}

label {
    font-weight: 500 !important;
}

.stTextInput > div > div > input {
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 16px;
}

.stSelectbox > div > div {
    border-radius: 10px;
}

.stButton > button {
    border-radius: 12px;
    height: 52px;
    font-size: 17px;
    font-weight: 600;
    width: 100%;
}

[data-testid="stMetric"] {
    background: #1f2937;
    padding: 20px;
    border-radius: 15px;
}

.stAlert {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model_data = joblib.load("churn_model.pkl")

    preprocessor = model_data["preprocessor"]
    model = model_data["model"]

    return preprocessor, model


try:

    preprocessor, model = load_model()

except Exception as e:

    st.error("❌ Unable to load the ML model.")
    st.error(f"Error: {e}")
    st.stop()


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">📊 Customer Retention Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict customer churn risk and identify customers who may need retention support.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# 👤 CUSTOMER INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )


with col2:

    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


# =========================================================
# 📱 PHONE & INTERNET SERVICES
# =========================================================

st.markdown(
    '<div class="section-title">📱 Phone & Internet Services</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )


with col2:

    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )


with col3:

    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


# =========================================================
# 📺 STREAMING SERVICES
# =========================================================

st.markdown(
    '<div class="section-title">📺 Streaming Services</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


with col1:

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )


with col2:

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# =========================================================
# 💳 BILLING & CONTRACT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">💳 Billing & Contract Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)


# ---------------------------------------------------------
# COLUMN 1
# ---------------------------------------------------------

with col1:

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )


# ---------------------------------------------------------
# COLUMN 2
# ---------------------------------------------------------

with col2:

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    tenure_input = st.text_input(
        "Tenure (months)",
        value="12",
        placeholder="Enter tenure in months"
    )


# ---------------------------------------------------------
# COLUMN 3
# ---------------------------------------------------------

with col3:

    monthly_charges_input = st.text_input(
        "Monthly Charges",
        value="50.00",
        placeholder="Enter monthly charges"
    )

    total_charges_input = st.text_input(
        "Total Charges",
        value="600.00",
        placeholder="Enter total charges"
    )


# =========================================================
# PREDICTION BUTTON
# =========================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Churn Risk",
    use_container_width=True
)


# =========================================================
# PREDICTION
# =========================================================

if predict_button:

    try:

        # -------------------------------------------------
        # CONVERT NUMERIC INPUTS
        # -------------------------------------------------

        tenure = float(tenure_input)
        monthly_charges = float(monthly_charges_input)
        total_charges = float(total_charges_input)


        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

        if tenure < 0 or tenure > 100:

            st.error(
                "❌ Tenure must be between 0 and 100 months."
            )

            st.stop()


        if monthly_charges < 0:

            st.error(
                "❌ Monthly Charges cannot be negative."
            )

            st.stop()


        if total_charges < 0:

            st.error(
                "❌ Total Charges cannot be negative."
            )

            st.stop()


        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        customer_data = pd.DataFrame([{

            "SeniorCitizen": senior_citizen,
            "gender": gender,
            "Partner": partner,
            "Dependents": dependents,

            "tenure": tenure,

            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,

            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,

            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,

            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges

        }])


        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        processed_data = preprocessor.transform(
            customer_data
        )


        # -------------------------------------------------
        # PREDICT
        # -------------------------------------------------

        churn_probability = model.predict_proba(
            processed_data
        )[0][1]

        churn_prediction = model.predict(
            processed_data
        )[0]


        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        if churn_probability < 0.30:

            risk_level = "LOW"

            risk_message = (
                "This customer currently has a low churn risk."
            )

        elif churn_probability < 0.70:

            risk_level = "MEDIUM"

            risk_message = (
                "This customer has a moderate churn risk."
            )

        else:

            risk_level = "HIGH"

            risk_message = (
                "This customer has a high churn risk "
                "and may require retention attention."
            )


        # =================================================
        # RESULTS
        # =================================================

        st.markdown(
            '<div class="section-title">📈 Churn Prediction</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)


        # -------------------------------------------------
        # CHURN PROBABILITY
        # -------------------------------------------------

        with col1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.1f}%"
            )


        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        with col2:

            if risk_level == "HIGH":

                st.error(
                    f"🔴 {risk_level} RISK"
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    f"🟠 {risk_level} RISK"
                )

            else:

                st.success(
                    f"🟢 {risk_level} RISK"
                )


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        with col3:

            prediction_text = (
                "Likely to Churn"
                if churn_prediction == "Yes"
                else "Likely to Stay"
            )

            st.metric(
                "Prediction",
                prediction_text
            )


        # -------------------------------------------------
        # RISK MESSAGE
        # -------------------------------------------------

        st.info(risk_message)


    # =====================================================
    # INVALID NUMERIC INPUT
    # =====================================================

    except ValueError:

        st.error(
            "❌ Please enter valid numbers for "
            "Tenure, Monthly Charges, and Total Charges."
        )


    # =====================================================
    # OTHER ERRORS
    # =====================================================

    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.error(
            f"Error: {e}"
        )