import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    This is an abstract base class for machine learning models.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        """
        Train the model.
        
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        Returns:
            self: The trained model
        """
        
        pass

class LinearRegressionModel(Model):
    """
    This class implements a linear regression model.
    """
    
    def train(self, X_train, y_train, **kwargs):
        """
        Train the linear regression model.
        
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
        
        Returns:
            self: The trained linear regression model
        """
        
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Linear regression model trained successfully.")
            return reg
        except Exception as e:
            logging.error(f"Error occurred during model training: {str(e)}")
            raise e