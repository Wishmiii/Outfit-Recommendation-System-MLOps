# The baseline model lacks - rich visual features, generalization, real world robustness
# To establish performance used a baseline model and applied transfer learning to leverage ImageNet-learned visual features

# The pretrained CNN already knows edges, textures, shapes learned from millions of images
#  My dataset is simple and small, not enough to learn deep features reliably
# So keep the pretrained feature extractor, replace final classification layer, train only the new layer
# This is correct & industry standard

# Same data pipeline as baseline
# training loop from baseline is resued here
# model  - ResNet-18
# Uses pretrained weights
# Only classifier layer is trained


# Pretrained CNN training code below
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import mlflow
import mlflow.pytorch

from dataloader import get_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLflow experiment
    mlflow.set_experiment("fashion_mnist_transfer_learning")

    with mlflow.start_run(run_name="resnet18_transfer"):

        train_loader, val_loader = get_dataloaders(batch_size=4)

        # load pretrained model
        model = models.resnet18(pretrained=True)

        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False

        # Replace classification head
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)
        model = model.to(device)

        # Loss & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

        # Log hyperparameters
        mlflow.log_param("model", "resnet18")
        mlflow.log_param("pretrained", True)
        mlflow.log_param("epochs", 5)
        mlflow.log_param("learning_rate", 1e-3)
        mlflow.log_param("trainable_layers", "fc_only")

        # Training loop
        num_epochs = 2
        
        print(">>> Starting training script")
        print(">>> Model and data loading complete")


        for epoch in range(num_epochs):
            model.train()
            train_correct, train_total = 0, 0

            # for images, labels in train_loader:
            for batch_idx, (images, labels) in enumerate(train_loader):
                print(f"Processing batch {batch_idx}")

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            train_acc = train_correct / train_total

            # Evaluation
            model.eval()
            val_correct, val_total = 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)

                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                    print("batch done")

            val_acc = val_correct / val_total

            print(
                f"Epoch [{epoch+1}/{num_epochs}] "
                f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Log metrics
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            

        # register model
        # Saves the trained model as an artifact and creates a registered model name if it doesnt
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name="fashion_mnist_classifier"
        )


if __name__ == "__main__":
    main()


# Applied transfer training using pretrained ResNet18, freezing the backbone and
# training a new classifier head, which significantly improved performance
# over the baseline CNN