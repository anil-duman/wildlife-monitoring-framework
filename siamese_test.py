import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import random

# === CONFIGURATION ===
image_size = (224, 224)
embedding_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === MODEL ===
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # THIS IS UPDATED: Set to V2 for compatibility with the trained model
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # As in your training code, adjust ResNet's last layer to our embedding_dim
        base_model.fc = nn.Linear(base_model.fc.in_features, embedding_dim)

        # Use base_model as feature_extractor, which is also consistent with your training code
        self.feature_extractor = base_model

    def forward_once(self, x):
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


# === TEST FUNCTIONS ===
def load_model(model_path):
    """Loads the trained model."""
    model = SiameseNetwork().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    # PyTorch's security warning regarding the pickle module.
    # It can be ignored for now if you trust the model's source.
    # Using 'weights_only=True' will be safer in the future.

    if 'model_state_dict' in checkpoint:
        # Check if the checkpoint contains the expected key
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
        if 'training_history' in checkpoint:
            print("Training history found in the checkpoint file.")
            return model, checkpoint['training_history']
    else:
        # If the checkpoint is directly the model's state_dict (legacy format)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully (legacy format)!")

    return model, None  # Returns None if training history is not found


def load_image(image_path, transform):
    """Loads and preprocesses a single image."""
    try:
        image = Image.open(image_path).convert("RGB")
        if transform:
            image = transform(image)
        return image.unsqueeze(0)  # Adds batch dimension
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def predict_similarity(model, img1_path, img2_path, transform):
    """Predicts similarity between two images."""
    model.eval()  # Sets the model to evaluation mode

    img1 = load_image(img1_path, transform)
    img2 = load_image(img2_path, transform)

    if img1 is None or img2 is None:
        return None, None

    img1, img2 = img1.to(device), img2.to(device)

    with torch.no_grad():  # Disables gradient calculation
        out1, out2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(out1, out2)
        similarity = 1 / (1 + distance)  # Converts distance to similarity (a simple heuristic)

    return distance.item(), similarity.item()


def test_on_directory(model, test_dir, transform, num_samples=100):
    """Tests the model on random pairs from the test directory."""
    print(f"Testing on directory: {test_dir}")

    # Get all classes and their images
    classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    class_to_images = {}

    for cls in classes:
        class_dir = os.path.join(test_dir, cls)
        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        if images:
            # Store full paths of valid images
            class_to_images[cls] = [os.path.join(class_dir, img) for img in images]

    print(f"Found {len(class_to_images)} classes.")

    # Generate test pairs
    test_pairs = []

    # Generate positive pairs (same class)
    # Aim for a certain number of positive pairs from each class
    for cls, images in class_to_images.items():
        if len(images) >= 2:
            # For simplicity, use random.sample
            for _ in range(min(num_samples // (len(class_to_images)), len(images) // 2)):
                img1, img2 = random.sample(images, 2)
                test_pairs.append((img1, img2, 1))  # 1 for similar

    # Generate negative pairs (different classes)
    # Aim to generate as many negative pairs as positive pairs
    num_negative_pairs_to_generate = len(test_pairs)
    for _ in range(num_negative_pairs_to_generate):
        # Must have at least two classes
        if len(class_to_images) < 2:
            break
        cls1, cls2 = random.sample(list(class_to_images.keys()), 2)
        img1 = random.choice(class_to_images[cls1])
        img2 = random.choice(class_to_images[cls2])
        test_pairs.append((img1, img2, 0))  # 0 for different

    random.shuffle(test_pairs)  # Shuffle the pairs
    print(f"Generated {len(test_pairs)} test pairs.")

    # Test the model
    predictions = []
    true_labels = []
    distances = []

    model.eval()
    with torch.no_grad():
        for img1_path, img2_path, true_label in tqdm(test_pairs, desc="Testing"):
            img1 = load_image(img1_path, transform)
            img2 = load_image(img2_path, transform)

            if img1 is None or img2 is None:
                # Skip this pair if images cannot be loaded
                continue

            img1, img2 = img1.to(device), img2.to(device)
            out1, out2 = model(img1, img2)
            distance = torch.nn.functional.pairwise_distance(out1, out2).item()

            distances.append(distance)
            true_labels.append(true_label)

            # Use a threshold to make a prediction (this can be adjusted)
            prediction = 1 if distance < 0.5 else 0  # 0.5 is the default threshold
            predictions.append(prediction)

    return predictions, true_labels, distances


def find_optimal_threshold(distances, true_labels):
    """Finds the optimal threshold for classification."""
    # Tries thresholds between min and max distances with 100 points
    if not distances:  # Check for empty list
        return 0.5, 0.0, []

    thresholds = np.linspace(min(distances) * 0.9, max(distances) * 1.1, 100)  # A wider range
    best_threshold = 0.5
    best_f1 = -1  # F1 score can be greater than 0

    threshold_results = []

    for threshold in thresholds:
        predictions = [1 if d < threshold else 0 for d in distances]
        f1 = f1_score(true_labels, predictions, zero_division=0)
        accuracy = accuracy_score(true_labels, predictions)

        threshold_results.append({
            'threshold': threshold,
            'f1': f1,
            'accuracy': accuracy
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1, threshold_results


def plot_test_results(distances, true_labels, predictions, threshold_results):
    """Visualizes the test results."""
    plt.figure(figsize=(18, 12))  # Larger figure size

    # Distance distribution
    plt.subplot(2, 3, 1)
    similar_distances = [d for d, label in zip(distances, true_labels) if label == 1]
    different_distances = [d for d, label in zip(distances, true_labels) if label == 0]

    plt.hist(similar_distances, bins=30, alpha=0.7, label='Similar Pairs', color='green')
    plt.hist(different_distances, bins=30, alpha=0.7, label='Different Pairs', color='red')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confusion Matrix
    plt.subplot(2, 3, 2)
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Different', 'Similar'],
                yticklabels=['Different', 'Similar'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Threshold vs F1 Score
    plt.subplot(2, 3, 3)
    thresholds = [r['threshold'] for r in threshold_results]
    f1_scores = [r['f1'] for r in threshold_results]
    accuracies = [r['accuracy'] for r in threshold_results]

    plt.plot(thresholds, f1_scores, 'b-', label='F1 Score', linewidth=2)
    plt.plot(thresholds, accuracies, 'r-', label='Accuracy', linewidth=2)
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Metrics Bar Chart
    plt.subplot(2, 3, 4)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [accuracy, precision, recall, f1]
    colors = ['blue', 'green', 'orange', 'red']

    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Test Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{value:.3f}', ha='center', va='bottom')

    # ROC-like curve (Distance vs True Positive Rate)
    plt.subplot(2, 3, 5)
    sorted_indices = np.argsort(distances)
    sorted_distances = np.array(distances)[sorted_indices]
    sorted_labels = np.array(true_labels)[sorted_indices]

    # Calculate TPR and FPR, handle empty arrays
    if np.sum(sorted_labels) > 0:
        tpr = np.cumsum(sorted_labels) / np.sum(sorted_labels)
    else:
        tpr = np.zeros_like(sorted_labels, dtype=float)

    if np.sum(1 - sorted_labels) > 0:
        fpr = np.cumsum(1 - sorted_labels) / np.sum(1 - sorted_labels)
    else:
        fpr = np.zeros_like(sorted_labels, dtype=float)

    plt.plot(fpr, tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-like Curve')
    plt.grid(True, alpha=0.3)

    # Visualization of sample predictions
    plt.subplot(2, 3, 6)
    if distances:  # Only sample if distances list is not empty
        sample_size = min(20, len(distances))
        sample_indices = random.sample(range(len(distances)), sample_size)

        sample_distances = [distances[i] for i in sample_indices]
        sample_labels = [true_labels[i] for i in sample_indices]
        sample_predictions = [predictions[i] for i in sample_indices]

        x_pos = range(len(sample_distances))
        colors = ['green' if pred == label else 'red'
                  for pred, label in zip(sample_predictions, sample_labels)]

        plt.scatter(x_pos, sample_distances, c=colors, alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.title('Sample Predictions (Green=Correct, Red=Wrong)')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, "No samples to visualize", horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def interactive_test(model, transform):
    """Interactive testing with two image paths."""
    print("\n" + "=" * 60)
    print("INTERACTIVE TESTING")
    print("=" * 60)

    while True:
        print("\nEnter two image paths to compare (or 'quit' to exit):")
        img1_path = input("Image 1 path: ").strip().strip('"')

        if img1_path.lower() == 'quit':
            break

        img2_path = input("Image 2 path: ").strip().strip('"')

        if img2_path.lower() == 'quit':
            break

        if not os.path.exists(img1_path) or not os.path.exists(img2_path):
            print("Error: One or both image paths don't exist!")
            continue

        distance, similarity = predict_similarity(model, img1_path, img2_path, transform)

        if distance is None:
            print("Error: Could not process images!")
            continue

        print(f"\nResults:")
        print(f"Distance: {distance:.4f}")
        print(f"Similarity: {similarity:.4f}")
        print(f"Prediction: {'SIMILAR' if distance < 0.5 else 'DIFFERENT'}")
        print("-" * 40)


def main():
    """Main test function."""
    print("=" * 60)
    print("SIAMESE NETWORK TEST")
    print("=" * 60)

    # Model and data paths
    model_path = "siamese_model_final.pth"  # Ensure this matches your trained model's save name
    test_dir = r"/path/to/data/test"  # Test directory

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please make sure you've trained the model first.")
        return

    # Load model
    print("Loading model...")
    model, training_history = load_model(model_path)

    # Define transform (must be same as training)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Test on directory if it exists
    if os.path.exists(test_dir):
        print(f"\nTesting on directory: {test_dir}")
        predictions, true_labels, distances = test_on_directory(model, test_dir, transform, num_samples=200)

        if predictions:  # If predictions list is not empty
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision = precision_score(true_labels, predictions, zero_division=0)
            recall = recall_score(true_labels, predictions, zero_division=0)
            f1 = f1_score(true_labels, predictions, zero_division=0)

            print("\n" + "=" * 40)
            print("TEST RESULTS")
            print("=" * 40)
            print(f"Total test samples: {len(predictions)}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

            # Find optimal threshold
            optimal_threshold, best_f1, threshold_results = find_optimal_threshold(distances, true_labels)
            print(f"\nOptimal threshold: {optimal_threshold:.4f}")
            print(f"Best F1 score: {best_f1:.4f}")

            # Plot results
            print("\nGenerating test plots...")
            plot_test_results(distances, true_labels, predictions, threshold_results)

        else:
            print("No valid test samples found!")

    else:
        print(f"Test directory '{test_dir}' not found!")
        print("You can still use interactive testing below.")

    # Interactive testing
    interactive_test(model, transform)


if __name__ == "__main__":
    main()