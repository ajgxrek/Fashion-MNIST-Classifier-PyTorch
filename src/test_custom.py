import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from src.model import FashionModel

def prepare_image(image_path):
    try:
        img = Image.open(image_path).convert('L')
        img = ImageOps.invert(img)
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        return transform(img).unsqueeze(0)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def run_prediction(image_path):
    classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    model = FashionModel()
    model.load_state_dict(torch.load("../models/model_ubrania.pth"))
    model.eval()

    input_tensor = prepare_image(image_path)
    if input_tensor is None: return

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item() * 100

    print(f"File: {image_path} | Prediction: {classes[prediction]} ({confidence:.2f}%)")

    plt.imshow(input_tensor.squeeze(), cmap="gray")
    plt.title(f"Predicted: {classes[prediction]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    run_prediction("../test_images/but.jpg")