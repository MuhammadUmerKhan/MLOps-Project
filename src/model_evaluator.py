import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):
    """
    This is an abstract base class for model evaluation.
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the evaluation scores.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        """
        pass

class MSE(Evaluation):
    """
    This class calculates the Mean Squared Error (MSE) for model evaluation.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Mean Squared Error (MSE) for model evaluation.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            mse (float): Mean Squared Error
        """
        try:
            logging.info("Calculating mean squared error")
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error occurred during MSE calculation: {str(e)}")
            raise e

class R2_score(Evaluation):
    """
    This class calculates the R-squared score (R2) for model evaluation.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the R-squared score (R2) for model evaluation.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            r2 (float): R-squared score
        """
        try:
            logging.info("Calculating R-squared score")
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"R-squared score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error occurred during R2 calculation: {str(e)}")
            raise e

class RMSE(Evaluation):
    """
    This class calculates the Root Mean Squared Error (RMSE) for model evaluation.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculate the Root Mean Squared Error (RMSE) for model evaluation.
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        
        Returns:
            rmse (float): Root Mean Squared Error
        """
        try:
            logging.info("Calculating root mean squared error")
            rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
            logging.info(f"Root Mean Squared Error: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error occurred during RMSE calculation: {str(e)}")
            raise e