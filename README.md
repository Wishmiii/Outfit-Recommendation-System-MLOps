# Fashion-MNIST Classifier with MLOps Pipeline

This project demonstrates an **end-to-end MLOps pipeline** for training, tracking, registering, and deploying a deep learning model using **PyTorch, MLflow, FastAPI, and Docker**.

The system classifies clothing images from the **Fashion-MNIST dataset** and exposes predictions through a REST API.

---

##  Project Overview

This project includes:

- Baseline CNN model
- Transfer Learning using ResNet18
- Experiment tracking with MLflow
- Model registry & versioning
- API deployment using FastAPI
- Containerization with Docker

---

## Model Approach

### Baseline Model
- Simple fully connected neural network
- Limited ability to learn complex visual features

### Transfer Learning Model
- Uses ResNet18 pretrained on ImageNet
- Freezes feature extractor
- Trains only final classification layer

 This improves:
- Generalization
- Feature extraction
- Performance on small datasets

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd closet_mlops

### 2. Create Virtual Environment
python -m venv venv

Activate it:

Windows:

venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Train the Model
Run transfer learning training:

python src/train_transfer.py

This will:
-Train the model
-Log experiments in MLflow
-Register model as:fashion_mnist_classifier

### 5. View MLFlow UI
mlflow ui

open in browser:
[http://127.](http://127.0.0.1:5000)

In MLflow UI:
Go to Models
Select fashion_mnist_classifier
Click the version (e.g., Version 1)
Click "Transition stage" → Production

 This step is required before running the API

### Run the API Locally
uvicorn api.main:app --reload

Open:

http://127.0.0.1:8000/docs

You will see the interactive API (Swagger UI)

### Test the API

Use the /predict endpoint:

Upload an image
Receive prediction:
{
  "predicted_class": "Sneaker",
  "confidence": 0.92
}
 Run with Docker
Build Image
docker build -t fashion-mlops-api .
Run Container
docker run -p 8000:8000 fashion-mlops-api

Then open:

http://localhost:8000/docs
```

 MLOps Workflow
   - Train model (train_transfer.py)
   - Log experiments using MLflow
   - Register model in MLflow registry
   - Promote model to Production
   - API dynamically loads production model
   - Serve predictions via FastAPI


 Key Skills Earned
   - Deep Learning (PyTorch)
   - Transfer Learning (ResNet18)
   - Experiment Tracking (MLflow)
   - Model Versioning & Registry
   - API Development (FastAPI)
   - Docker Containerization
   - End-to-End ML Pipeline Design


 Notes
    Ensure model is in Production stage before running API
    MLflow uses local SQLite by default
    First training run may take longer due to model download


 Author

Wishmi Subasinghe
BSc (Hons) Artificial Intelligence and Data Science


