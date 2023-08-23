import os
from clearml import Task, Dataset
import pandas as pd
from config import *
from pathlib2 import Path
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
task = Task.init(
    project_name=EXPERIMENT_NAME,
    task_name="feature_engineering",
    task_type="data_processing",
    reuse_last_task_id=False
)

feature_engineering = "data/preprocessing"
if not os.path.exists(feature_engineering): os.makedirs(feature_engineering)

dataset = Dataset.get(
    dataset_project=EXPERIMENT_NAME,
    dataset_name= "loan_approval_v2"
)

local_instance = dataset.get_local_copy()
print("Data processing dataset id: ",dataset.id)

dataset_df = pd.read_csv(Path(local_instance)/"loan.csv")
data = dataset_df.fillna({'Gender': 'Not Known', 'Married':'Not Known', 'Dependents': -1, 'Self_Employed': 'Not Known', 'LoanAmount':-999,'Loan_Amount_Term':-999.0,'Credit_History':-1.0})
labels = data['Gender'].value_counts().keys().tolist()
values = data['Gender'].value_counts().tolist()

#Plotting EDA
colors = ['Yellow', 'Green', 'Black']
fig = go.Figure(data=[go.Pie(labels=labels,values=values)])
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=25,
                      marker=dict(colors=colors, line=dict(color='#000000', width=1)))
fig.update_layout(title_text='Genderwise Loan Application')
fig = px.bar(data, x="Gender", y="LoanAmount",
                 color='Loan_Status', barmode='group',
                 height=400)
fig.update_layout(title_text='LoanAmount, Loan Status approved Genderwise')
fig = px.bar(data, x="Property_Area", y="LoanAmount", color="Self_Employed", barmode="group")

target = data["Loan_Status"].map({"N":0,"Y":1}).astype(int)
features = data.loc[:,data.columns!="Loan_Status"]
features["Dependents"]=features["Dependents"].replace("3+","3")
features["Dependents"]=features['Dependents'].fillna(0).astype(int)
features = pd.get_dummies(features,columns=["Gender","Married","Education","Self_Employed","Property_Area"])
columns_to_scale = ["ApplicantIncome","CoapplicantIncome","Dependents","LoanAmount","Loan_Amount_Term"]
scaler = StandardScaler()
scaler.fit(features[columns_to_scale])
dump(scaler,feature_engineering+"/selected_scaled.joblib")
pd.DataFrame(columns_to_scale,columns=["scaled_features"]).to_csv("data/preprocessing/scaled_features_list.csv", index=False)
x_train,x_test,y_train,y_test = train_test_split(features.drop(["Loan_ID"],axis=1),target,test_size=0.2,random_state=42)
features.to_csv(path_or_buf=feature_engineering+"/features.csv",index=False)
target.to_csv(feature_engineering+"/target.csv",index=False)

x_train.to_csv(feature_engineering+"/x_train.csv",index=False)
x_test.to_csv(feature_engineering+"/x_test.csv",index=False)
y_train.to_csv(feature_engineering+"/y_train.csv",index=False)
y_test.to_csv(feature_engineering+"/y_test.csv",index=False)

new_instance_data = Dataset.create(
    dataset_name="preprocessed_split_data",
    dataset_project=dataset.project
)

new_instance_data.add_files(feature_engineering)
new_instance_data.finalize(auto_upload=True)

