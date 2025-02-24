import logging
import pandas as pd
from zenml import step

class IngestData:
    """
    Ingests data from a data_path.
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path (str): path to the data
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from the data_path

        """
        logging.info(f"Ingesting data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        return df

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a specified path.

    Args:
        data_path (str): The path to the data file.

    Returns:
        pd.DataFrame: The loaded data.
    """
    try:
        ingest_data = IngestData(data_path)
        data = ingest_data.get_data()
        return data
    except Exception as e:
        logging.info(f"Error while ingesting data: {e}")
        raise e