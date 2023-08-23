import datetime
from collections import OrderedDict

import pandas as pd
import shap
import streamlit as st
from matplotlib import pyplot as plt



st.set_page_config(layout="wide")
"""
if not model.is_model_trained():
    raise Exception("The credit scoring model has not been trained. Please run run.py.")
"""

def get_loan_request():
    Gender = st.sidebar.radio("Select your Gender",options=["Male","Female"])
    Married = st.sidebar.radio("Are you married", options=["Yes", "No"])
    Dependents = st.sidebar.slider("No of Dependents", 0, 10, 0)
    ApplicantIncome = st.sidebar.slider("Applicant Income", 0, 99999, 0)
    CoapplicantIncome = st.sidebar.slider("Co-Applicant Income", 0, 99990, 0)
    Education = st.sidebar.radio(
        "Are you a Graduate ?",
        [
            "Graduate",
            "Not Graduate",
        ],
    )

    LoanAmount = st.sidebar.slider("Loan amount", 0, 99999, 0)
    Loan_Amount_Term= st.sidebar.slider("Preferred Tenure in weeks", 0, 500, 24, step=1)
    Property_Area = st.sidebar.radio("your Residence property",options=["Urban","Semiurban","Rural"])
    Self_Employed = st.sidebar.radio("Self Employed", options=["Yes", "No"])
    Credit_History = st.sidebar.radio("Self attested credit history", options=[1, 0])
    submit = st.sidebar.button("Submit")
    if submit:
        return OrderedDict(
        {
            "Gender":[Gender],
            "Married": [Married],
            "Dependents": [Dependents],
            "Education": [Education],
            "Self_Employed": [Self_Employed],
            "ApplicantIncome": [ApplicantIncome],
            "CoapplicantIncome": [CoapplicantIncome],
            "LoanAmount": [LoanAmount],
            "Loan_Amount_Term": [Loan_Amount_Term],
            "Credit_History": [Credit_History],
            "Property_Area": [Property_Area],
        }
    )


# Application
st.title("Loan Application")

# Input Side Bar
st.header("User input:")
loan_request = get_loan_request()
df = pd.DataFrame.from_dict(loan_request)
print(loan_request)

# Full feature vector
st.header("Feature vector (user input + zipcode features + user features):")
vector = model._get_online_features_from_feast(loan_request)
ordered_vector = loan_request.copy()
key_list = vector.keys()
key_list = sorted(key_list)
for vector_key in key_list:
    if vector_key not in ordered_vector:
        ordered_vector[vector_key] = vector[vector_key]
df = pd.DataFrame.from_dict(ordered_vector)


# Results of prediction
st.header("Application Status (model prediction):")
result = model.predict(loan_request)

if result == 0:
    st.success("Your loan has been approved!")
elif result == 1:
    st.error("Your loan has been rejected!")


# Feature importance
st.header("Feature Importance")
X = pd.read_parquet("data/training_dataset_sample.parquet")
explainer = shap.TreeExplainer(model.classifier)
shap_values = explainer.shap_values(X)
left, mid, right = st.columns(3)
with left:
    plt.title("Feature importance based on SHAP values")
    shap.summary_plot(shap_values[1], X)
    st.set_option("deprecation.showPyplotGlobalUse", False)
    st.pyplot(bbox_inches="tight")
    st.write("---")

with mid:
    plt.title("Feature importance based on SHAP values (Bar)")
    shap.summary_plot(shap_values, X, plot_type="bar")
    st.pyplot(bbox_inches="tight")
