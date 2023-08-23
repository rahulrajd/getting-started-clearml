from clearml import Task, Dataset
from config import *
import pandas as pd
data_task = Task.init(
    project_name=EXPERIMENT_NAME,
    task_name= "ingest_data",
    task_type="data_processing",
    reuse_last_task_id=False
)

dataset = Dataset.create(
    dataset_name="loan_approval_v2",
    dataset_project=EXPERIMENT_NAME
)
dataset_df = pd.read_csv("data/loan.csv")
dataset.add_files("data/")
dataset.get_logger().report_table(title="Loan Approval Dataset",series="head",table_plot=dataset_df.head())
dataset.finalize(auto_upload=True)
print("Dataset ID",dataset.id)
