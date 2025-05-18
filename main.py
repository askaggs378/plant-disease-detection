import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os

from models.vit_model import build_vit
from utils.preprocessing import get_data_loaders

# Configuration
DATA_DIR = "data/Subset"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "outputs/best_model.pth"

print("Starting script...")

if __name__ == "__main__":
    try:
        # Load data
        print("Loading data...")
        train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, IMAGE_SIZE, BATCH_SIZE)
        num_classes = len(class_names)

        # Load model
        print("Building model...")
        model = build_vit(num_classes)
        model = model.to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_accuracy = 0.0

        # Training loop
        for epoch in range(EPOCHS):
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            model.train()
            train_loss = 0.0
            correct = 0

            for images, labels in tqdm(train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()

            train_acc = correct / len(train_loader.dataset)
            print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

            # Validation
            model.eval()
            val_correct = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()

            val_acc = val_correct / len(val_loader.dataset)
            print(f"Validation Accuracy: {val_acc:.4f}")

            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                os.makedirs("outputs", exist_ok=True)
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"Saved new best model with accuracy: {val_acc:.4f}")

    except Exception as e:
        print("Error occurred:", e)

