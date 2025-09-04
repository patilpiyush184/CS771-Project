import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from transformers import ViTModel, ViTFeatureExtractor
import numpy as np

# Custom Dataset class for handling image data
class CustomImageDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx] if self.labels is not None else -1  # Use -1 for unlabeled data
        if self.transform:
            image = self.transform(image)
        return image, label


# Load ViT Feature Extractor for preprocessing
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Define transformations to preprocess input images for ViT
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Load Pretrained Vision Transformer model
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
vit_model.eval()


def extract_features(model, dataloader, device="cpu"):
    """
    Extract features from images using the Vision Transformer.
    Args:
        model: The pretrained Vision Transformer model.
        dataloader: DataLoader for input data.
        device: Device for computation ('cpu' or 'cuda').
    Returns:
        Tuple of extracted features and corresponding labels.
    """
    features, labels = [], []
    model.to(device)
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            outputs = model(images).last_hidden_state[:, 0, :]  # CLS token embeddings
            features.append(outputs.cpu())
            labels.append(lbls)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    return features, labels


def compute_prototypes(features, labels, num_classes=10):
    """
    Compute class prototypes by averaging feature embeddings for each class.
    Args:
        features: Feature embeddings.
        labels: Ground truth labels.
        num_classes: Number of classes.
    Returns:
        Tensor of prototypes, one per class.
    """
    prototypes = []
    for cls in range(num_classes):
        class_features = features[labels == cls]
        if len(class_features) > 0:
            prototype = class_features.mean(dim=0)  # Mean of features for the class
        else:
            prototype = torch.zeros(features.shape[1])  # Default prototype for empty classes
        prototypes.append(prototype)
    return torch.stack(prototypes)


def classify_with_prototypes(prototypes, query_features):
    """
    Classify query features using prototypes based on Euclidean distance.
    Args:
        prototypes: Class prototypes.
        query_features: Feature embeddings of query samples.
    Returns:
        Predicted labels for the query samples.
    """
    distances = torch.cdist(query_features, prototypes)  # Compute distances
    predicted_labels = distances.argmin(dim=1)  # Class with the closest prototype
    return predicted_labels


def pseudo_label_and_update(features, prototypes, confidence_threshold=0.8):
    """
    Generate pseudo-labels and select confident samples for updating prototypes.
    Args:
        features: Feature embeddings of unlabeled data.
        prototypes: Current prototypes.
        confidence_threshold: Minimum confidence required to consider a pseudo-label.
    Returns:
        Confident features and their pseudo-labels.
    """
    distances = torch.cdist(features, prototypes)
    confidences, pseudo_labels = torch.softmax(-distances, dim=1).max(dim=1)

    # Select samples with high confidence
    confident_indices = confidences > confidence_threshold
    confident_features = features[confident_indices]
    confident_labels = pseudo_labels[confident_indices]

    return confident_features, confident_labels


def train_sequential_models():
    """
    Train models sequentially using multiple datasets while mitigating forgetting.
    Returns:
        Accuracy matrix where rows correspond to models and columns to evaluation datasets.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    momentum = 0.9  # Momentum for prototype updates
    num_classes = 10  # Number of classes in the dataset
    model_prototypes = []  # Store prototypes for each model
    accuracies = []  # Accuracy matrix

    # Load and preprocess the first dataset (D1)
    D1 = torch.load("dataset/part_one_dataset/train_data/1_train_data.tar.pth")
    train_images, train_labels = D1["data"], D1["targets"]
    train_dataset = CustomImageDataset(train_images, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Extract features and compute initial prototypes for D1
    train_features, train_lbls = extract_features(vit_model, train_loader, device)
    prototypes = compute_prototypes(train_features, train_lbls, num_classes)
    model_prototypes.append(prototypes)

    # Sequentially process and update prototypes for datasets D2 to D10
    for i in range(2, 11):
        # Load next dataset (Di)
        Di = torch.load(f"dataset/part_one_dataset/train_data/{i}_train_data.tar.pth")
        images = Di["data"]
        dataset = CustomImageDataset(images, transform=transform)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        # Extract features and pseudo-labels
        features, _ = extract_features(vit_model, loader, device)
        confident_features, confident_labels = pseudo_label_and_update(features, prototypes)

        # Update prototypes using pseudo-labeled samples
        new_prototypes = compute_prototypes(confident_features, confident_labels, num_classes)
        prototypes = momentum * prototypes + (1 - momentum) * new_prototypes  # Momentum-based update

        model_prototypes.append(prototypes)

    # Evaluate models on held-out datasets
    for model_idx, prototypes in enumerate(model_prototypes):
        model_accuracies = []
        for eval_idx in range(1, model_idx + 2):
            # Load evaluation dataset
            eval_data = torch.load(f"dataset/part_one_dataset/eval_data/{eval_idx}_eval_data.tar.pth")
            eval_images, eval_labels = eval_data["data"], eval_data["targets"]
            eval_dataset = CustomImageDataset(eval_images, eval_labels, transform=transform)
            eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

            # Extract features and evaluate using current prototypes
            eval_features, eval_lbls = extract_features(vit_model, eval_loader, device)
            predicted_labels = classify_with_prototypes(prototypes, eval_features)

            # Calculate accuracy
            accuracy = round((predicted_labels == eval_lbls).float().mean().item(), 2)
            model_accuracies.append(accuracy)
            print(f"Model f{model_idx+1} accuracy on D̂{eval_idx}: {accuracy * 100:.2f}%")

        accuracies.append(model_accuracies)
    return accuracies


accuracy_matrix = train_sequential_models()

print("\nAccuracy Matrix:")
for row in accuracy_matrix:
    print(row)
