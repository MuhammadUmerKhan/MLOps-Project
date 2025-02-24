import logging
from abc import ABC, abstractmethod
from typing import Union
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    This is an abstract base class for data handling strategies.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    This class handles data preprocessing by dropping unnecessary columns, filling missing values, and converting categorical variables to numerical ones.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1,
            )
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            # write "No review" in review_comment_message column
            data["review_comment_message"].fillna("No review", inplace=True)

            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(f"Error occurred during data preprocessing: {str(e)}")
            raise e

class DataSplitStrategy(DataStrategy):
    """
    This class splits the data into training and testing sets.
    """
    
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Splits the data into training and testing sets.
        
        Args:
            data (pd.DataFrame): The input data.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training and testing sets.
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error(f"Error occurred during data splitting: {str(e)}")
            raise e

class DataCleaning:
    """
    This class handles data cleaning by applying different data handling strategies.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy
    
    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle the data
        """
        
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"Error occurred during data handling: {str(e)}")
            raise e

if __name__ == "__main__":
    data = pd.read_csv("/home/muhammadumerkhan/MLOps-Project/data/olist_customers_dataset.csv")
    data_cleaner = DataCleaning(data, DataPreProcessStrategy())
    data_cleaner.handle_data()