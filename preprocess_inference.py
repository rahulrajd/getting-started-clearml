import xgboost as xgb
import pandas as pd
import numpy as np
import json
from collections import OrderedDict

class Preprocess():
    def __init__(self):
        pass

    def preprocess(self,body:OrderedDict):
        ordered_payload={}
        _key= body.keys()
        for key in _key:
            if key not in ordered_payload:
                body[key] = body[key]
        df = pd.DataFrame.from_dict(ordered_payload)




    def postprocess(self):
        pass


if __name__ == "__main__":
    payload = OrderedDict([('Gender', ['Male']), ('Married', ['Yes']), ('Dependents', [0]), ('Education', ['Graduate']),
                           ('Self_Employed', ['Yes']),
                           ('ApplicantIncome', [0]), ('CoapplicantIncome', [0]), ('LoanAmount', [0]), ('Loan_Amount_Term', [24]), ('Credit_History', [1]), ('Property_Area', ['Urban'])])
    #print(payload.keys())

    data = pd.read_csv("data/loan.csv")

    from category_encoders import OneHotEncoder
