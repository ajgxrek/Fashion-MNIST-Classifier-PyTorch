import torch
import torch.nn as nn
import torch.optim as optim
from src.data_setup import get_dataloaders
from src.model import FashionModel

def train_and_validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_dataloaders()
    model = FashionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"  Test Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "../models/model_ubrania.pth")
    print("Model saved as model_ubrania.pth")

if __name__ == "__main__":
    train_and_validate()