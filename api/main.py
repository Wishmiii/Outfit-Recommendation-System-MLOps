from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

# for loading models
from src.model_loader import load_model

# This file is for serving predictions over HTTP

# 1. App initialization
app = FastAPI(title="Fashion-MNIST Classifier API")


# 2. Load model ONCE at startup
# loads a model
# not tied to a specific model - mlflow decides which verion is production
# you can promote a version to production if needed
MODEL_NAME = "fashion_mnist_classifier"


model = load_model(MODEL_NAME, alias="production")

# 3. Class labels
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# 4. Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 5. Prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    return {
        "predicted_class": CLASS_NAMES[predicted_class.item()],
        "confidence": round(confidence.item(), 4)
    }

# This deploys the trained model as a FastAPI service, loading it from the 
# MLflow model registry at startup and serving predictions via a REST endpoint
