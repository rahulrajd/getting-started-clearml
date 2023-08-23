from platform import node
from clearml import Task
from clearml.automation import PipelineController
from config import *

def compare_metrics_and_publish_best(**kwargs):
    from clearml import OutputModel
    # Keep track of best node details
    current_best = dict()

    # For each incoming node, compare against current best
    for node_name, training_task_id in kwargs.items():
        # Get the original task based on the ID we got from the pipeline
        task = Task.get_task(task_id=training_task_id)
        accuracy = task.get_reported_scalars()['Performance']['Accuracy']['y'][0]
        model_id = task.get_models()['output'][0].id
        # Check if accuracy is better than current best, if so, overwrite current best
        if accuracy > current_best.get('accuracy', 0):
            current_best['accuracy'] = accuracy
            current_best['node_name'] = node_name
            current_best['model_id'] = model_id
            print(f"New current best model: {node_name}")

    # Print the final best model details and log it as an output model on this step
    print(f"Final best model: {current_best}")
    OutputModel(name="best_pipeline_model", base_model_id=current_best.get('model_id'), tags=['pipeline_winner'])

if __name__ == "__main__":
    pipe = PipelineController(
        name=PIPELINE_NAME,
        project=EXPERIMENT_NAME,
        version='0.0.1'
    )

    pipe.set_default_execution_queue('CPU Queue')
    pipe.add_parameter('training_seeds', [42, 420, 500])
    pipe.add_parameter('query_date', '2022-01-01')

    pipe.add_step(
        name='ingest_data',
        base_task_project=EXPERIMENT_NAME,
        base_task_name='ingest_data',
    )
    pipe.add_step(
        name='feature_engineering',
        parents=['ingest_data'],
        base_task_project=EXPERIMENT_NAME,
        base_task_name='feature_engineering',
    )
    training_nodes = []
    for i, random_state in enumerate(pipe.get_parameters()['training_seeds']):
        node_name = f'model_training_{i}'
        training_nodes.append(node_name)
        pipe.add_step(
            name=node_name,
            parents=['feature_engineering'],
            base_task_project=EXPERIMENT_NAME,
            base_task_name='model_training',
            parameter_override={'General/num_boost_round': 250,
                                'General/test_size': 0.5,
                                'General/random_state': random_state}
        )

    pipe.add_function_step(
        name='select_best_model',
        parents=training_nodes,
        function=compare_metrics_and_publish_best,
        function_kwargs={node_name: '${%s.id}' % node_name for node_name in training_nodes},
        monitor_models=["best_pipeline_model"]
    )


    # for debugging purposes use local jobs
    pipe.start_locally(run_pipeline_steps_locally=True)
    # Starting the pipeline (in the background)
    #pipe.start()

    print('Done!')