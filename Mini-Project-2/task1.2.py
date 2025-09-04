"""
Important --

Please download the f10_prototypes file from here:
https://drive.google.com/file/d/1u6OqG9OZGTWdrytMzhAExX4v9H1TPo4V/view?usp=sharing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import ViTModel, ViTFeatureExtractor
import numpy as np


# Custom Dataset Class
class CustomImageDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx] if self.labels is not None else -1
        if self.transform:
            image = self.transform(image)
        return image, label


# Feature Extraction Function
def extract_features(model, dataloader, device="cpu"):
    """Extract features using the Vision Transformer."""
    features, labels = [], []
    model.to(device)
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = model(images).last_hidden_state[:, 0, :]  # CLS token
            features.append(outputs.cpu())
            labels.append(lbls)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


# Compute Prototypes Function
def compute_prototypes(features, labels, num_classes=10):
    """Compute class prototypes."""
    prototypes = []
    for cls in range(num_classes):
        class_features = features[labels == cls]
        if len(class_features) > 0:
            prototype = class_features.mean(dim=0)
        else:
            prototype = torch.zeros(features.shape[1])
        prototypes.append(prototype)
    return torch.stack(prototypes)


# Pseudo-labeling and Update Function
def pseudo_label_and_update(features, prototypes, confidence_threshold=0.8):
    """Generate pseudo-labels and select samples for training."""
    distances = torch.cdist(features, prototypes)
    confidences, pseudo_labels = torch.softmax(-distances, dim=1).max(dim=1)

    # Select samples with high confidence
    confident_indices = confidences > confidence_threshold
    confident_features = features[confident_indices]
    confident_labels = pseudo_labels[confident_indices]

    return confident_features, confident_labels


# Classification Function
def classify_with_prototypes(prototypes, query_features):
    """Classify using prototypes by calculating Euclidean distance."""
    distances = torch.cdist(query_features, prototypes)
    predicted_labels = distances.argmin(dim=1)
    return predicted_labels


# Vision Transformer Initialization
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()


# Task 2 Training and Evaluation
def train_sequential_models_task2():
    """
    Train models sequentially for datasets D11 to D20 with distribution shifts.
    Starts with f10_prototypes from Task 1 and adapts prototypes for subsequent datasets.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    momentum = 0.9  # Momentum for prototype updates
    num_classes = 10
    model_prototypes = []  # List to store prototypes for each model iteration
    accuracies = []  # To store accuracy values for evaluation datasets

    # Load f10_prototypes (final prototypes from Task 1)
    f10_prototypes = torch.load('f10_prototypes.pth')
    model_prototypes.append(f10_prototypes)

    for i in range(11, 21):
        # Load the training dataset for the current task (Di)
        Di = torch.load(f"dataset/part_two_dataset/train_data/{i}_train_data.tar.pth")
        images = Di["data"]
        dataset = CustomImageDataset(images, transform=transform)  # Wrap data in a Dataset object
        loader = DataLoader(dataset, batch_size=16, shuffle=False)  # Load data in batches

        # Extract features for the current dataset using the pretrained Vision Transformer
        features, _ = extract_features(vit_model, loader, device)

        # Generate pseudo-labels and filter confident samples based on the existing prototypes (f10_prototypes)
        confident_features, confident_labels = pseudo_label_and_update(features, f10_prototypes)

        # Compute new prototypes based on the pseudo-labeled data
        new_prototypes = compute_prototypes(confident_features, confident_labels, num_classes)

        # Update prototypes with momentum to balance adaptation and stability
        f10_prototypes = momentum * f10_prototypes + (1 - momentum) * new_prototypes

        # Store the updated prototypes for later evaluation
        model_prototypes.append(f10_prototypes)

    # Evaluate the models on held-out datasets (D̂1 to D̂20)
    for model_idx, prototypes in enumerate(model_prototypes, start=2):  # Start indexing from f11
        model_accuracies = []  # Accuracies for the current model

        # Evaluate on datasets D̂1 to D̂(model_idx + 10)
        for eval_idx in range(1, model_idx + 10):
            if eval_idx == 21:  # Skip index 21 as it doesn't exist
                continue

            # Load evaluation dataset based on eval_idx
            if eval_idx < 11:
                eval_data = torch.load(f"dataset/part_one_dataset/eval_data/{eval_idx}_eval_data.tar.pth")
            else:
                eval_data = torch.load(f"dataset/part_two_dataset/eval_data/{eval_idx-10}_eval_data.tar.pth")

            eval_images, eval_labels = eval_data["data"], eval_data["targets"]
            eval_dataset = CustomImageDataset(eval_images, eval_labels, transform=transform)  # Wrap in Dataset
            eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)  # Load in batches

            # Extract features for evaluation
            eval_features, eval_lbls = extract_features(vit_model, eval_loader, device)

            # Classify samples using the current prototypes
            predicted_labels = classify_with_prototypes(prototypes, eval_features)

            # Compute accuracy for the evaluation dataset
            accuracy = (predicted_labels == eval_lbls).float().mean().item()
            model_accuracies.append(accuracy)

            # Print the accuracy for the current evaluation dataset
            print(f"Model f{model_idx+9} accuracy on D̂{eval_idx}: {accuracy * 100:.2f}%")

        # Store the accuracies for the current model
        accuracies.append(model_accuracies)

    return accuracies


# Run the training and evaluation process for Task 2
accuracy_matrix_task2 = train_sequential_models_task2()

# Print the accuracy matrix for Task 2
print("\nAccuracy Matrix for Task 2:")
for row in accuracy_matrix_task2:
    print(row)
