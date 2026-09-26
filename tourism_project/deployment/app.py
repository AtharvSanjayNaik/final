import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

MODEL_REPO = "ASNaik/tourism-prediction-model"
MODEL_FILE = "best_tourism_prediction_model_v1.joblib"

# Download and load the model
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(model_path)

# Streamlit UI for Wellness Tourism Package Prediction
st.title("Wellness Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the
Wellness Tourism Package, based on their profile and past interaction data.
Please enter the customer details below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, value=35)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0)
Occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Average Number of Trips per Year", min_value=0.0, max_value=20.0, value=3.0)
Passport = st.selectbox("Holds a Valid Passport?", [0, 1])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
OwnCar = st.selectbox("Owns a Car?", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (below 5 yrs)", min_value=0.0, max_value=5.0, value=0.0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", min_value=0.0, max_value=100000.0, value=20000.0)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome,
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Will Purchase the Package" if prediction == 1 else "Will Not Purchase the Package"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
