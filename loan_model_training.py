from clearml import Task,Dataset
from config import *
import pandas as pd
from pathlib2 import Path
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
from joblib import load, dump
task = Task.init(
    project_name=EXPERIMENT_NAME,
    task_name="model_training",
    task_type="training",
    output_uri=True
)

task.set_base_docker(docker_image="python:3.7")

# Training args
training_args = {
    'eval_metric': "rmse",
    'objective': 'reg:squarederror',
    'test_size': 0.2,
    'random_state': 42,
    'num_boost_round': 100
}
task.connect(training_args)
local_path = Dataset.get(
    dataset_name='preprocessed_split_data',
    dataset_project=EXPERIMENT_NAME
).get_local_copy()
local_path = Path(local_path)
X_train = pd.read_csv(local_path/"x_train.csv")

X_test = pd.read_csv(local_path / "x_test.csv")
y_test = pd.read_csv(local_path / "y_test.csv")
y_train = pd.read_csv(local_path / "y_train.csv")
features = pd.read_csv(local_path / 'features.csv')
target = pd.read_csv(local_path / 'target.csv')

# Split data
#X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=training_args['test_size'], random_state=training_args['random_state'])
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

bst = xgb.train(
    training_args,
    dtrain,
    num_boost_round=training_args['num_boost_round'],
    evals=[(dtrain, "train"), (dtest, "test")],
    verbose_eval=0
)

bst.save_model("best_model")
plot_importance(bst)
plt.show()

preds = bst.predict(dtest)
predictions = [round(value) for value in preds]
accuracy = accuracy_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print(f"Model trained with accuracy: {accuracy} and recall: {recall}")
# Save the actual accuracy as an artifact so we can get it as part of the pipeline
task.get_logger().report_scalar(
    title='Performance',
    series='Accuracy',
    value=accuracy,
    iteration=0
)
task.get_logger().report_scalar(
    title='Performance',
    series='Recall',
    value=recall,
    iteration=0
)
