import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Load the scaler and models
try:
    scaler_path = os.path.join(script_dir, 'scaler.pkl')
    lr_model_path = os.path.join(script_dir, 'logistic_regression_model_bank.pkl')
    svc_model_path = os.path.join(script_dir, 'svc_model_bank.pkl')
    knn_model_path = os.path.join(script_dir, 'knn_model_bank.pkl')

    scaler = joblib.load(scaler_path)
    lr_model = joblib.load(lr_model_path)
    svc_model = joblib.load(svc_model_path)
    knn_model = joblib.load(knn_model_path)
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'scaler.pkl', 'logistic_regression_model_bank.pkl', 'svc_model_bank.pkl', and 'knn_model_bank.pkl' are in the same directory as the script.")
    st.stop()

# Define a function to preprocess user input
def preprocess_input(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Separate numerical and categorical columns
    numerical_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = input_df.select_dtypes(include=['object']).columns

    # Apply one-hot encoding to categorical columns
    X_categorical_df = pd.get_dummies(input_df[categorical_cols], drop_first=True)

    # Reindex to ensure all columns from training data are present.
    # This requires having access to the columns from the training data.
    # A robust solution would be to save the list of columns after training.
    # For now, we'll infer the columns from the original categorical columns
    # and assume the order and presence of numerical columns are consistent.

    # Get the list of columns from the original training data's processed features
    # This assumes X_processed from the training notebook is available or can be reconstructed
    # based on the original data and preprocessing steps.
    # A more robust approach would be to save X_processed.columns during training.
    # For this example, we'll recreate the column list based on the original data's categorical columns.
    original_categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    all_categorical_dummies = pd.get_dummies(pd.DataFrame(columns=original_categorical_cols), drop_first=True)

    # Ensure the processed categorical input has the same columns as the training data
    X_categorical_df = X_categorical_df.reindex(columns=all_categorical_dummies.columns, fill_value=0)


    # Scale numerical columns
    X_numerical_scaled = scaler.transform(input_df[numerical_cols])
    X_numerical_scaled_df = pd.DataFrame(X_numerical_scaled, columns=numerical_cols)

    # Concatenate processed numerical and categorical features
    X_processed = pd.concat([X_numerical_scaled_df.reset_index(drop=True), X_categorical_df.reset_index(drop=True)], axis=1)

    # Ensure the final processed input DataFrame has the same columns as the training data
    # This is the most crucial step to prevent the ValueError.
    # We need the exact column order and names from the training data (X_train or X_processed).
    # Since we don't have X_processed.columns readily available in this script,
    # we'll need to infer the column order based on the preprocessing steps.
    # A better approach would be to save X_processed.columns during training and load it here.

    # For now, we'll try to reconstruct the expected column order based on the numerical
    # columns and the sorted one-hot encoded categorical columns.
    expected_columns = list(numerical_cols) + sorted(all_categorical_dummies.columns)
    X_processed = X_processed.reindex(columns=expected_columns, fill_value=0)


    return X_processed

# Streamlit App
st.title("Bank Customer Churn Prediction")

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

    # Make predictions with each model
    prediction_lr = lr_model.predict(processed_input)
    prediction_svc = svc_model.predict(processed_input)
    prediction_knn = knn_model.predict(processed_input)

    # Display predictions
    st.subheader("Prediction Results:")
    st.write(f"Logistic Regression Prediction: {'Yes' if prediction_lr[0] == 'yes' else 'No'}")
    st.write(f"SVC Prediction: {'Yes' if prediction_svc[0] == 'yes' else 'No'}")
    st.write(f"KNN Prediction: {'Yes' if prediction_knn[0] == 'yes' else 'No'}")
