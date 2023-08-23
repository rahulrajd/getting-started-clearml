import datetime
from collections import OrderedDict
import pandas as pd
import streamlit as st
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



# Results of prediction
st.header("Application Status (model prediction):")
result = 0

if result == 0:
    st.success("Your loan has been approved!")
elif result == 1:
    st.error("Your loan has been rejected!")


