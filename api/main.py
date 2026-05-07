from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

from src.model_loader import load_model

app = FastAPI(title="Fashion-MNIST Classifier API")

MODEL_NAME = "fashion_mnist_classifier"
model = load_model(MODEL_NAME, alias="production")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fashion Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f4f6f8;
                margin: 0;
                padding: 40px;
            }
            .container {
                max-width: 650px;
                margin: auto;
                background: white;
                padding: 30px;
                border-radius: 16px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            }
            h1 {
                text-align: center;
                color: #1f2937;
            }
            p {
                text-align: center;
                color: #6b7280;
            }
            input {
                margin-top: 20px;
                width: 100%;
            }
            button {
                margin-top: 20px;
                width: 100%;
                padding: 12px;
                border: none;
                border-radius: 10px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                cursor: pointer;
            }
            button:hover {
                background: #1d4ed8;
            }
            img {
                display: none;
                max-width: 100%;
                margin-top: 20px;
                border-radius: 12px;
            }
            .result {
                margin-top: 25px;
                padding: 18px;
                border-radius: 12px;
                background: #ecfdf5;
                color: #065f46;
                font-size: 18px;
                display: none;
            }
            .small {
                font-size: 13px;
                color: #9ca3af;
                margin-top: 25px;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fashion-MNIST Classifier</h1>
            <p>Upload a clothing image and the model will predict the category.</p>

            <input type="file" id="fileInput" accept="image/*">
            <img id="preview">

            <button onclick="predict()">Predict</button>

            <div class="result" id="result"></div>

            <div class="small">
                Powered by FastAPI, PyTorch, MLflow and Docker
            </div>
        </div>

        <script>
            const fileInput = document.getElementById("fileInput");
            const preview = document.getElementById("preview");
            const result = document.getElementById("result");

            fileInput.addEventListener("change", () => {
                const file = fileInput.files[0];
                if (file) {
                    preview.src = URL.createObjectURL(file);
                    preview.style.display = "block";
                    result.style.display = "none";
                }
            });

            async function predict() {
                const file = fileInput.files[0];

                if (!file) {
                    alert("Please choose an image first.");
                    return;
                }

                const formData = new FormData();
                formData.append("file", file);

                result.style.display = "block";
                result.innerHTML = "Predicting...";

                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();

                result.innerHTML = `
                    <strong>Prediction:</strong> ${data.predicted_class}<br>
                    <strong>Confidence:</strong> ${(data.confidence * 100).toFixed(2)}%
                `;
            }
        </script>
    </body>
    </html>
    """


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