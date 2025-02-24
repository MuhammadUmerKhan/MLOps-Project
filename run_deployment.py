from pipelines.deployment_pipeline import continous_deployment_pipeline
import click
from rich import print
from typing import cast
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY = "deploy"
PREDICT = "predict"
DEPLOY_AND_PREDICT = "deploy_and_predict"

@click.command()
@click.option(
    "--config",  # ✅ Removed incorrect alias "c"
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    help="Optionally choose to only run the deployment pipeline (`deploy`), "
         "only run a prediction (`predict`), or run both (`deploy_and_predict`)."
)
@click.option(
    "--min-accuracy",
    default=0.5,
    help="Minimum accuracy required to deploy the model"
)
def run_deployment(config: str, min_accuracy: float):
    """Runs the deployment pipeline based on the given configuration."""
    
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    
    if deploy:
        continous_deployment_pipeline(
            data_path="/home/muhammadumerkhan/MLOps-Project/data/olist_customers_dataset.csv",
            min_accuracy=min_accuracy,
            workers=3,
            timeout=60
        )

    if predict:
        print(
            "You can run:\n "
            f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}'"
            "[/italic green]\n ...to inspect your experiment runs within the MLflow UI.\n"
        )
        
    # Fetch existing services with the same pipeline name, step name, and model name
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run:\n"
                f"[italic green]zenml model-deployer models delete {str(service.uuid)}[/italic green]"
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is running. Run the deployment pipeline first with `--config deploy`."
        )

if __name__ == "__main__":
    run_deployment()
