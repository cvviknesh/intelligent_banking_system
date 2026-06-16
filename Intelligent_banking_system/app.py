import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sqlalchemy import create_engine, text
import pymysql
from dotenv import load_dotenv
import os

# --- Load Environment Variables ---
load_dotenv(dotenv_path="sql.env")

# --- Secure Database Connection ---
def get_db_connection():
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    database = os.getenv("DB_NAME")

    engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}")
    return engine.connect()

# --- Save Loan Prediction Result ---
def save_loan_prediction(account_number, age, income, loan_amount, credit_score, loan_term, employment_type, marital_status, prediction):
    try:
        session = get_db_connection()
        query = text("""
        INSERT INTO loan_predictions 
        (account_number, age, income, loan_amount, credit_score, loan_term, employment_type, marital_status, prediction)
        VALUES (:account_number, :age, :income, :loan_amount, :credit_score, :loan_term, :employment_type, :marital_status, :prediction)
        """)
        session.execute(query, {
            "account_number": account_number,
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "loan_term": loan_term,
            "employment_type": employment_type,
            "marital_status": marital_status,
            "prediction": prediction
        })
        session.commit()  # Ensure the transaction is saved
        print("✅ Prediction successfully saved to database.")
        return True
    except Exception as e:
        st.error(f"❌ Error saving loan prediction: {e}")
        print(f"Error saving loan prediction: {e}")
        return False

# --- Save Customer Segmentation Result ---
def save_customer_segment(age, income, credit_score, segment, products):
    try:
        session = get_db_connection()
        query = text("""
        INSERT INTO customer_segments (age, income, credit_score, segment, products_recommended)
        VALUES (:age, :income, :credit_score, :segment, :products)
        """)
        session.execute(query, {
            "age": age,
            "income": income,
            "credit_score": credit_score,
            "segment": segment,
            "products": ", ".join(products)
        })
        session.commit()  # Ensure the transaction is saved
        print("✅ Customer segmentation successfully saved to database.")
        return True
    except Exception as e:
        st.error(f"❌ Error saving customer segment: {e}")
        print(f"Error saving customer segment: {e}")
        return False

# --- STREAMLIT APP CONFIG ---
st.set_page_config(page_title="Intelligent Banking System", page_icon="🏦", layout="wide")

# --- Set Background Image ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://www.shutterstock.com/image-illustration/application-laptop-business-graph-analytics-260nw-1935848740.jpg");
        background-size: cover;
        background-position: center center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session State Defaults ---
for key in ["age", "income", "credit_score"]:
    if key not in st.session_state:
        st.session_state[key] = 25 if key == "age" else 50000 if key == "income" else 600

# --- CLUSTER FUNCTION ---
def get_cluster_and_profile(credit_score):
    if 300 <= credit_score < 450:
        return 3, "Bad"
    elif 450 <= credit_score < 550:
        return 2, "Fair"
    elif 550 <= credit_score < 700:
        return 1, "Good"
    else:
        return 0, "Excellent"

# --- DATA LOADING ---
selected_features = ['Age', 'Income', 'CreditScore']
train_data = pd.read_csv('Bank_data.csv')[selected_features]
scaler = StandardScaler()
scaled_train_data = scaler.fit_transform(train_data)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(scaled_train_data)

with open("model.pkl", "rb") as file:
    loan_default_model = pickle.load(file)

# --- Sidebar ---
st.sidebar.title("🏦 Intelligent Banking System")
menu = st.sidebar.radio(
    "Select a feature:",
    ["🏦 Home", "📉 Loan Default Prediction", "📊 Customer Segmentation & 🎯 Product Recommendation","📋 View Saved Records"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("🔹 **Developed by Sowmiya Lakshmee, Jagadish, Abishek, Viknesh** | 📊 **AI-Powered Banking Solutions** 🚀")

# --- Home ---
if menu == "🏦 Home":
    st.title("💰 Intelligent Banking System")
    st.markdown("""
        ### AI-powered banking insights for:
        - 🔹 **Loan Default Prediction**
        - 🔹 **Customer Segmentation**
        - 🔹 **Product Recommendations**
    """)
    st.success("🚀 Welcome! Select a feature from the left menu to get started.")

# --- View Saved Records ---
if menu == "📋 View Saved Records":
    st.title("📋 View Saved Records")
    with st.spinner("Fetching records..."):
        try:
            engine = create_engine(f"mysql+pymysql://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}")
            loan_df = pd.read_sql("SELECT * FROM loan_predictions", con=engine)
            if loan_df.empty:
                st.warning("No Loan Predictions found.")
            else:
                st.dataframe(loan_df)
        except Exception as e:
            st.error(f"❌ Error fetching loan predictions: {e}")
            print(f"Error fetching loan predictions: {e}")

        st.markdown("---")

        # View Customer Segments
        st.subheader("📊 Customer Segments & 🎯 Recommendations")
        try:
            segment_df = pd.read_sql("SELECT * FROM customer_segments", con=engine)
            if segment_df.empty:
                st.warning("No Customer Segments found.")
            else:
                st.dataframe(segment_df)
        except Exception as e:
            st.error(f"❌ Error fetching customer segments: {e}")
            print(f"Error fetching customer segments: {e}")


# --- Loan Default Prediction ---
elif menu == "📉 Loan Default Prediction":
    st.title("📉 Loan Default Prediction")
    st.write("### Predict whether a customer is at risk of defaulting on a loan.")

    col1, col2, col3 = st.columns(3)
    with col1:
        account_number = st.text_input("🆔 Account Number (15–16 digits)")
        if account_number and not (len(account_number) in [15, 16] and account_number.isdigit()):
            st.warning("⚠️ Account number should be 15 or 16 digits long.")
        st.session_state.age = st.number_input("🎂 Age", min_value=18, max_value=60, value=st.session_state.age)
        st.session_state.income = st.number_input("💰 Income", min_value=15000, value=st.session_state.income)

    with col2:
        loan_amount = st.number_input("🏦 Loan Amount", min_value=0, value=10000)
        credit_score_category = st.selectbox("📊 Cibil Score Category", 
                                             ["300-450", "450-550", "550-700", "700+"])
        credit_score_mapping = {"300-450": 375, "450-550": 500, "550-700": 625, "700+": 750}
        st.session_state.credit_score = credit_score_mapping[credit_score_category]

         # External link under CIBIL Score
        st.markdown("[ℹ️ Calculate your CIBIL Score](https://moneyview.in/credit-score/login)")

        loan_term = st.selectbox("📆 Loan Term (months)", [12, 24, 36, 48, 60])

    with col3:
        employment_type = st.selectbox("💼 Employment Type", ["Full-time", "Part-time", "Selfemployed", "Unemployed"])
        marital_status = st.selectbox("❤️ Marital Status", ["Single", "Married", "Divorced"])

    if st.button("🚀 Predict Loan Default"):
        # Convert categorical values to numeric
        employment_mapping = {"Full-time": 0, "Part-time": 1, "Selfemployed": 2, "Unemployed": 3}
        marital_mapping = {"Single": 0, "Married": 1, "Divorced": 2}

        employment_numeric = employment_mapping[employment_type]
        marital_numeric = marital_mapping[marital_status]

        # Prepare the input data with numeric values only
        input_data = np.array([[st.session_state.age, 
                                st.session_state.income, 
                                loan_amount,
                                st.session_state.credit_score, 
                                loan_term,
                                employment_numeric, 
                                marital_numeric]])

        # Make the prediction
        prediction = loan_default_model.predict(input_data)
        
        if prediction[0] == 0:
            st.success("✅ No Loan Default Risk Detected! (Reliable Customer)")
            prediction_text = "No Default"
        else:
            st.error("❌ High Risk of Loan Default! Take Precaution.")
            prediction_text = "Default"
    
        st.info(f"📌 Account Number: **{account_number}**")
        
        # Save Prediction to Database
        if save_loan_prediction(account_number, st.session_state.age, st.session_state.income, 
                                loan_amount, st.session_state.credit_score, loan_term, 
                                employment_type, marital_status, prediction_text):
            st.success("✅ Prediction saved to database.")
        else:
            st.error("❌ Failed to save prediction to database.")

# --- Customer Segmentation ---
elif menu == "📊 Customer Segmentation & 🎯 Product Recommendation":
    st.title("📊 Customer Segmentation & 🎯 Product Recommendation")
    st.write("### Classify customers based on income & spending behavior.")

    customer_age = st.number_input("🎂 Age", min_value=18, max_value=100, value=st.session_state.age)
    customer_income = st.number_input("💰 Income", min_value=1000, max_value=1000000, value=st.session_state.income)
    customer_credit_score = st.number_input("📊 Cibil Score", min_value=300, max_value=850, value=st.session_state.credit_score)

    if st.button("🔍 Predict Customer Segment & 🎯 Recommendation Product"):
        new_customer = pd.DataFrame({
            'Age': [customer_age],
            'Income': [customer_income],
            'CreditScore': [customer_credit_score],
        })
        new_customer_scaled = scaler.transform(new_customer)
        predicted_cluster = kmeans.predict(new_customer_scaled)[0]

        _, profile = get_cluster_and_profile(customer_credit_score)
        credit_score_to_cluster = {"Bad": 3, "Fair": 2, "Good": 1, "Excellent": 0}
        predicted_cluster = credit_score_to_cluster[profile]
        
        st.success(f"🧾 **Segment Prediction:** {profile} Customer")
        st.info(f"🔐 Assigned to Cluster **{predicted_cluster}**")

        product_recommendations = {
            3: ['Savings Account', 'Checking Account', 'Insurance Policy', 'Emergency Loan', 'Debt Consolidation Loan'],
            2: ['Personal Loan', 'Auto Loan', 'Travel Loan', 'Insurance Policy', 'Healthcare Savings'],
            1: ['Home Loan', 'Education Savings Plan', 'Mutual Fund', 'Retirement Savings', 'Stock Investment'],
            0: ['Wealth Management', 'International Investment', 'Real Estate Investment', 'Pension Plan', 'Property Investment']
        }

        recommended_products = product_recommendations.get(predicted_cluster, [])
        st.markdown("### 🎁 Recommended Financial Products:")
        for product in recommended_products:
            st.write(f"🔹 {product}")

        # Save Segmentation to Database
        if save_customer_segment(customer_age, customer_income, customer_credit_score, profile, recommended_products):
            st.success("✅ Customer segmentation saved to database.")
        else:
            st.error("❌ Failed to save segmentation to database.")

# --- Footer ---
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: right; font-size: 14px; color: white; font-weight: bold;
                padding: 10px; background-color: #0e1117;'>
        Developed by Sowmiya Lakshmee, Jagadish, Abishek, Viknesh | 📊 AI-Powered Banking Solutions 🚀
    </div>
    """,
    unsafe_allow_html=True
)



