from pipelines.training_pipeline import training_pipeline
from zenml.client import Client

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path="/home/muhammadumerkhan/MLOps-Project/data/olist_customers_dataset.csv")

# This command is used to launch the MLflow tracking UI with the experiment data stored by ZenML.
# mlflow ui --backend-store-uri file:/home/muhammadumerkhan/.config/zenml/local_stores/ab17e0b3-5cf3-4964-9e2e-f578d4399dc1/mlruns