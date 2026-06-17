import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Load trained model
model = load_model("notebook/model.h5")

#  Load the encoder and the standard scaler 


with open("notebook/label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)

with open("notebook/one_hot_encoder_geography.pkl","rb") as file:
    one_hot_encoder_geography = pickle.load(file)

with open("notebook/standard_scaler.pkl","rb") as file:
    standard_scaler = pickle.load(file)


# Streamlit app

st.title("Customer Churn Prediction")

# User Input

credit_score = st.number_input("Credit Score", min_value=0, max_value=1000)
geography  =st.selectbox("Geography", one_hot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", min_value=18, max_value=100)
tenure = st.slider("Tenure", min_value=0, max_value=10)
balance = st.number_input("Balance", min_value=0.0)
num_of_products = st.slider("Number of Products", 1, 4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0)



# Prepare input data

input_data_df = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_credit_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

# Preprocess input data

geo_encoded =  one_hot_encoder_geography.transform([input_data_df["Geography"]]).toarray()
geo_encoded_df =  pd.DataFrame(geo_encoded,columns =one_hot_encoder_geography.get_feature_names_out(["Geography"]))

input_data_df = input_data_df.drop("Geography", axis=1)

# concatenate 
input_data_final= pd.concat([input_data_df.reset_index(drop=True),geo_encoded_df],axis =1)
input_data_final["Gender"] = label_encoder_gender.transform(input_data_final["Gender"])


#  Scaled the data
input_data_scaled = standard_scaler.transform(input_data_final)



# Display prediction
if st.button("Predict"):
    # Predict Churn 
    prediction = model.predict(input_data_final)
    prediction_probability = prediction[0][0]

    st.write(f"Churn Probability: {prediction_probability:.2%}")

    if prediction_probability > 0.5:
        st.error("Customer is likely to churn.")
    else:
        st.success("Customer is likely to stay.")


# Streamlit run app.py