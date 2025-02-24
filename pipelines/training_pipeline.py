from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.model_train import train_model
from steps.config import ModelNameConfig

@pipeline(enable_cache=False)
def training_pipeline(data_path: str):
    data = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(data)
    
    config = ModelNameConfig()
    model = train_model(X_train, X_test, y_train, y_test, config=config)
    
    r2_score, rsme = evaluate_model(model, X_test, y_test)