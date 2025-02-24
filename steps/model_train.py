import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from zenml import step
from zenml.client import Client
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from steps.config import ModelNameConfig
from mlflow.models.signature import infer_signature

# Get the experiment tracker from the active ZenML stack
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None, output_materializers={"model": SklearnMaterializer})  # ✅ Assign materializer
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig
) -> RegressorMixin:
    """
    Trains a linear regression model with MLflow experiment tracking.

    Returns:
        RegressorMixin: Trained model
    """

    # ✅ Ensure there is no active MLflow run before starting a new one
    if mlflow.active_run():
        mlflow.end_run()

    model = None

    if config.ml_model_name == "LinearRegression":
        with mlflow.start_run():
            mlflow.sklearn.autolog()
            
            model = LinearRegressionModel()
            trained_model = model.train(X_train, y_train)

            # ✅ Infer model signature
            signature = infer_signature(X_train, trained_model.predict(X_train))

            # ✅ Log the trained model with MLflow
            mlflow.sklearn.log_model(trained_model, artifact_path="linear_regression_model", signature=signature)

            logging.info("Model trained and logged to MLflow successfully.")
            return trained_model  # ✅ Now uses SklearnMaterializer
        
    else:
        logging.error(f"Unsupported model name: {config.ml_model_name}")
        raise ValueError("Unsupported model name")
