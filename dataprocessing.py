import clearml
from clearml import Task
from clearml import Dataset
from clearml.automation import PipelineDecorator
from clearml import TaskTypes
from clearml.automation import PipelineController
import json
from clearml.logger import Logger

def load_data(url,file_name):
    import pandas as pd
    from clearml import Dataset
    load_data = Dataset.create(dataset_name="loan_approval_dataset", dataset_project="loan_approval",
                                  dataset_tags=["classification"])
    load_data.add_files(path=url)

    id = load_data.id
    data = pd.read_csv(url+file_name)
    load_data.upload()
    load_data.finalize()
    return data,id


def explore_data(data_frame,id,url):
    from clearml import Dataset
    import plotly.express as px
    import plotly.graph_objects as go
    data = data_frame.fillna({'Gender': 'Not Known', 'Married':'Not Known', 'Dependents': -1, 'Self_Employed': 'Not Known', 'LoanAmount':999,'Loan_Amount_Term':999.0,'Credit_History':-1.0})
    featurize_data = Dataset.create(dataset_project="loan_approval",dataset_name="loan_approval_dataset",parent_datasets=[id],dataset_tags=["classification"])
    data.to_csv(url+"/null_removed_data.csv",index=False)
    labels = data['Gender'].value_counts().keys().tolist()
    values = data['Gender'].value_counts().tolist()
    colors = ['Yellow', 'Green', 'Black']
    fig = go.Figure(data=[go.Pie(labels=labels,
                                 values=values)])
    fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=25,
                      marker=dict(colors=colors, line=dict(color='#000000', width=1)))
    fig.update_layout(title_text='Genderwise Loan Application')
    featurize_data.get_logger().report_plotly(title="Gender wise loan application",series="value",figure=fig)
    featurize_data.add_files(path=url)
    fig = px.bar(data, x="Gender", y="LoanAmount",
                 color='Loan_Status', barmode='group',
                 height=400)
    fig.update_layout(title_text='LoanAmount, Loan Status approved Genderwise')
    featurize_data.get_logger().report_plotly(title="LoanAmount, Loan Status approved Genderwise", series="value", figure=fig)
    fig = px.bar(data, x="Property_Area", y="LoanAmount", color="Self_Employed", barmode="group")
    featurize_data.get_logger().report_plotly(title="Self Employed People laon application for the property area", series="value",
                                              figure=fig)
    featurize_data.upload()
    featurize_data.finalize()
    id = featurize_data.id
    return data, id

def engineer_and_spilt(data_frame,id,url):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd
    from joblib import dump
    from clearml import Dataset
    #Encoding
    data = data_frame.copy()
    engineer_handler = Dataset.create(dataset_project="loan_approval",dataset_name="loan_approval_dataset",parent_datasets=[id],dataset_tags=["classification"])
    target = data["Loan_Status"].map({"N":0,"Y":1}).astype(int)
    features = data.loc[:,data.columns!="Loan_Status"]
    features = pd.get_dummies(features,columns=["Gender","Married","Dependents","Education","Self_Employed","Property_Area"])
    columns_to_scale = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
    #engineer_handler.get_logger().report_table(title="Columns selected to scale",series="columns",table_plot=columns_to_scale)
    scaler = StandardScaler()
    scaler.fit(features[columns_to_scale])
    dump(scaler,"data/selected_scaled.joblib")
    pd.DataFrame(columns_to_scale,columns=["scaled_features"]).to_csv("data/preprocessing/scaled_features_list.csv", index=False)
    features[columns_to_scale] = scaler.transform(features[columns_to_scale])


    #train and test split
    x_train,x_test,y_train,y_test = train_test_split(features.drop(["Loan_ID"],axis=1),target,test_size=0.2,random_state=42)

    dump(x_train,"data/x_train.joblib")
    dump(x_test, "data/x_test.joblib")
    dump(y_train, "data/y_train.joblib")
    dump(y_test, "data/y_test.joblib")
    engineer_handler.add_files(path=url)
    engineer_handler.upload()
    engineer_handler.finalize()

if __name__ == "__main__":

    pipe = PipelineController(name="data_processing",
                              project="loan_approval",
                              version="1.0")
    pipe.add_tags("loan_approval")
    pipe.add_function_step(
        name="data_loading",
        function=load_data,
        function_kwargs= dict(url="data/",file_name="loan.csv"),
        function_return=["data","id"],
        cache_executed_step=True
    )

    pipe.add_function_step(
        name="explore_data",
        function=explore_data,
        function_kwargs=dict(data_frame='${data_loading.data}',id='${data_loading.id}',url="data/"),
        function_return = ["data","id"]
    )
    pipe.add_function_step(
        name="preprocess_data",
        function=engineer_and_spilt,
        function_kwargs=dict(data_frame='${explore_data.data}', id='${explore_data.id}', url="data/"),
    )

    pipe.set_default_execution_queue('default')
    #pipe.start()
    pipe.start_locally(run_pipeline_steps_locally=True)
    pipe.stop()
