import logging
import pandas as pd
from zenml import step
from src.data_cleanner import DataCleaning, DataPreProcessStrategy, DataSplitStrategy
from typing_extensions import Annotated
from typing import Tuple

@step
def clean_data(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """
    Performs data cleaning and splitting.

    Args:
        data (pd.DataFrame): The input data.
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleanning = DataCleaning(data, process_strategy)
        processed_data = data_cleanning.handle_data()
        
        split_strategy = DataSplitStrategy()
        data_cleaning = DataCleaning(processed_data, split_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        
        logging.info("Data cleaning and splitting completed successfully.")
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(f"Error occurred during data cleaning and splitting: {str(e)}")
        raise e