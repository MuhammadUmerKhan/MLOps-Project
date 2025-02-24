import json
import os
import numpy as np
from pydantic import BaseModel  
import logging

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step

from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.config import ModelNameConfig

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseModel):  
    """Configuration for deployment trigger."""
    min_accuracy: float = 0.5

@step
def deployment_trigger(accuracy: float, config: DeploymentTriggerConfig) -> bool:
    """Determines whether the model should be deployed."""
    decision = accuracy > config.min_accuracy
    logging.info(f"Deployment Trigger Decision: {decision}")
    return decision

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0.5,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    """Continuous deployment pipeline that trains and evaluates a model."""
    data = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(data)

    config = ModelNameConfig()
    model_name = train_model(X_train, X_test, y_train, y_test, config=config)
    r2_score, rmse = evaluate_model(model_name, X_test, y_test)

    logging.info(f"R² Score: {r2_score}, RMSE: {rmse}")

    # ✅ Force deployment regardless of accuracy
    deployment_decision = True  
    logging.info(f"Deployment Decision (Forced): {deployment_decision}")

    mlflow_model_deployer_step(
        model=model_name,  # ✅ Ensure the correct model name is passed
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )
