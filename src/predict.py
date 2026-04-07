import torch
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.data_setup import get_dataloaders
from src.model import FashionModel


def predict_batch(num_images=20):
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    train_loader, _ = get_dataloaders(batch_size=num_images)

    images, labels = next(iter(train_loader))

    model = FashionModel().to(device)
    model.load_state_dict(torch.load("../models/model_ubrania.pth", map_location=device))
    model.eval()

    with torch.no_grad():
        images_dev = images.to(device)
        outputs = model(images_dev)
        _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(12, 10))
    for i in range(num_images):
        plt.subplot(4, 5, i + 1)
        img = images[i].squeeze()
        plt.imshow(img, cmap="gray")

        is_correct = preds[i] == labels[i]
        color = "green" if is_correct else "red"

        plt.title(f"P: {classes[preds[i]]}\nL: {classes[labels[i]]}", color=color, fontsize=9)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    predict_batch(20)