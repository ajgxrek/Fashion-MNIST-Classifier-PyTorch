# Fashion-MNIST Image Classification with PyTorch

This repository contains a modular implementation of a neural network classifier trained on the **Fashion-MNIST** dataset. The project demonstrates a complete machine learning pipeline, including data preprocessing, model architecture definition, validation logic, and inference on external images.

## Performance and Results

The model utilizes a Multi-Layer Perceptron (MLP) architecture and achieves approximately **87% accuracy** on the test set.

### Batch Prediction Visualization

The following visualization shows the model's predictions on a random batch from the test set. Green labels indicate correct classifications, while red labels indicate errors:

*(Place your screenshot file path here, e.g., `docs/batch_results.png`)*

---

## Project Structure

The codebase is organized into functional modules to ensure maintainability and clarity:

* **model.py**: Contains the `FashionModel` class definition, utilizing Linear layers with ReLU activation and Dropout for regularization.
* **data_setup.py**: Manages dataset downloading and the creation of `DataLoader` objects with standard normalization.
* **train.py**: The primary execution script for training, featuring validation loops and model state saving.
* **predict.py**: A utility script for visualizing model performance on random samples from the test distribution.
* **test_custom.py**: An inference script designed to test the model's performance on external image files such as `.jpg` or `.png`.

---

## Case Study: Generalization and Structural Limitations

A significant component of this project involved evaluating the model's robustness against **high-contrast custom images** that differ from the original training distribution.

### Observation
An image of a boot on a solid black background was classified as a **T-shirt with 100% confidence**.

### Technical Analysis
* **Spatial Invariance**: Because the architecture relies strictly on **Fully Connected (Linear) layers**, it is highly sensitive to the global distribution of pixel intensities. It lacks spatial invariance, meaning it cannot effectively recognize shapes regardless of their specific pixel coordinates.
* **Feature Overlap**: In the downsampled 28x28 grayscale space, the high-intensity mass of the boot mathematically aligned with the pixel "footprint" associated with the T-shirt class in the training data.
* **Engineering Conclusion**: This result highlights the necessity of **Convolutional Neural Networks (CNNs)** for computer vision tasks. CNNs utilize filters to detect local features such as edges and textures, providing the hierarchical pattern recognition required to handle variations in object scale and orientation.

---

## Roadmap for Improvement

The current version serves as a baseline (MLP). Future iterations will focus on the following enhancements to reach production-level performance:

### 1. CNN Implementation
Transitioning from a Multi-Layer Perceptron (MLP) to a **Convolutional Neural Network (CNN)**. Unlike linear layers, CNNs use convolutional filters to capture spatial hierarchies, allowing the model to recognize patterns like edges, textures, and shapes regardless of their position in the image.

### 2. Data Augmentation
To improve the model's robustness against real-world photo variations, I plan to introduce a data augmentation pipeline during the training phase using `torchvision.transforms`. This includes:
* **Random Rotations**: Teaching the model to recognize objects even if they are not perfectly upright.
* **Scaling and Cropping**: Ensuring the model is invariant to the size and framing of the object.
* **Horizontal Flips**: Doubling the dataset's diversity for symmetrical items.

