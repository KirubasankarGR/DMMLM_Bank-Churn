import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the scaler and KNN model
try:
    scaler = joblib.load('scaler.pkl')
    knn_model = joblib.load('knn_model_bank.pkl')
except FileNotFoundError:
    st.error("Scaler or KNN model file not found. Please ensure 'scaler.pkl' and 'knn_model_bank.pkl' are in the same directory.")
    st.stop()

# Define a function to preprocess user input
def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Separate numerical and categorical columns
    numerical_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = input_df.select_dtypes(include=['object']).columns

    # Apply one-hot encoding to categorical columns (handle potential missing columns by reindexing)
    X_categorical_df = pd.get_dummies(input_df[categorical_cols], drop_first=True)
    # Reindex to ensure all columns from training data are present
    # This requires knowing the columns from the training data preprocessing step.
    # For simplicity, we'll assume the columns generated during training are available or can be inferred.
    # A more robust approach would be to save the list of columns during training.
    # For this example, we'll create dummy columns for all possible categories if they don't exist
    # based on the original dataset columns (excluding the target 'y').
    original_categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    all_categorical_dummies = pd.get_dummies(pd.DataFrame(columns=original_categorical_cols), drop_first=True)
    X_categorical_df = X_categorical_df.reindex(columns = all_categorical_dummies.columns, fill_value=0)

    # Scale numerical columns
    X_numerical_scaled = scaler.transform(input_df[numerical_cols])
    X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols)

    # Concatenate processed numerical and categorical features
    X_processed = pd.concat([X_numerical_scaled_df, X_categorical_df], axis=1)

    return X_processed

# Streamlit App
st.title("Bank Customer Churn Prediction (KNN Model)")

st.header("Enter Customer Details:")

# Input fields for user
age = st.slider("Age", 18, 92, 30)
job = st.selectbox("Job", ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
marital = st.selectbox("Marital Status", ['married', 'single', 'divorced'])
education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])
default = st.selectbox("Credit Default", ['no', 'yes'])
balance = st.number_input("Balance", value=0.0)
housing = st.selectbox("Housing Loan", ['no', 'yes'])
loan = st.selectbox("Personal Loan", ['no', 'yes'])
contact = st.selectbox("Contact Communication Type", ['unknown', 'cellular', 'telephone'])
day = st.slider("Last Contact Day of Month", 1, 31, 15)
month = st.selectbox("Last Contact Month of Year", ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
duration = st.number_input("Last Contact Duration (seconds)", value=0)
campaign = st.number_input("Number of Contacts During This Campaign", value=1)
pdays = st.number_input("Number of Days Since Last Contact", value=-1)
previous = st.number_input("Number of Contacts Before This Campaign", value=0)
poutcome = st.selectbox("Outcome of the Previous Marketing Campaign", ['unknown', 'other', 'failure', 'success'])

# Create a dictionary with user inputs
input_features = {
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'balance': balance,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'day': day,
    'month': month,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome
}

if st.button("Predict Churn"):
    # Preprocess the input data
    processed_input = preprocess_input(input_features)

    # Make prediction with the KNN model
    prediction_knn = knn_model.predict(processed_input)

    # Display prediction
    st.subheader("Prediction Result (KNN Model):")
    st.write(f"Prediction: {'Yes' if prediction_knn[0] == 'yes' else 'No'}")
