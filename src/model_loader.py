# to load the correct model from MLflow

import mlflow.pytorch
import torch

def load_model(model_name:str, alias:str = "production"):
    """Load a pytorch model from the MLflow model registry"""
    
    model_uri = f"models:/{model_name}@{alias}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model
