import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple
from src.model_evaluator import MSE, R2_score, RMSE
from sklearn.base import RegressorMixin
import mlflow
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker # experimental tracker

@step(experiment_tracker=experiment_tracker.name if experiment_tracker else None)  # Handle None case
def evaluate_model(model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
    ) -> Tuple[
        Annotated[float, "r2_score"],
        Annotated[float, "rmse_score"]
    ]:
    """
    Evaluates the trained model using Mean Squared Error (MSE), R-squared score (R2), and Root Mean Squared Error (RMSE).

    Args:
        model (RegressorMixin): Trained model
        X_test (pd.DataFrame): Testing data
        y_test (pd.Series): Testing labels
    Returns:
        r2_score (float): R-squared score
        rmse_score (float): Root Mean Squared Error score
    """
    try:
        prediction = model.predict(X_test)
        
        mse_eval = MSE()
        mse_score = mse_eval.calculate_scores(y_test.values, prediction)
        mlflow.log_metric("MSE", mse_score)
        
        r2_eval = R2_score()
        r2_score = r2_eval.calculate_scores(y_test.values, prediction)
        mlflow.log_metric('r2_score', r2_score)
        
        rmse_eval = RMSE()
        rmse_score = rmse_eval.calculate_scores(y_test.values, prediction)
        mlflow.log_metric('RMSE', rmse_score)
        
        return r2_score, rmse_score
    except Exception as e:
        logging.error(f"Error occurred during model evaluation: {str(e)}")
        raise e