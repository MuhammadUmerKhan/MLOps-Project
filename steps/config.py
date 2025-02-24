from pydantic import BaseModel

class ModelNameConfig(BaseModel):  
    """
    Configuration for the model name.
    """
    ml_model_name: str = "LinearRegression"

    class Config:
        protected_namespaces = ()  # ✅ Avoids namespace conflicts
