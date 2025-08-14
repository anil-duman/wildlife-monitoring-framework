import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageFile
import os
import random
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Allow PIL to load corrupted images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === CONFIGURATION ===
image_size = (224, 224)
embedding_dim = 128 # Embedded vector size
epochs = 10
batch_size = 32
learning_rate = 0.001
margin = 1.0 # Margin value for contrastive loss

# SAMPLING CONFIGURATION FOR LARGE DATASETS
MAX_IMAGES_PER_CLASS = 100  # Maximum number of images per class (dataset size)
MAX_POSITIVE_PAIRS_PER_CLASS = 200  # Maximum number of positive pairs per class
MAX_NEGATIVE_PAIRS_PER_CLASS = 200  # Maximum number of negative pairs per class

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# === DATASET ===
class FastSiameseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_images = {}

        print("Loading and sampling images...")
        # Sample images from each class
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):
                all_images = []
                for f in os.listdir(class_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_path = os.path.join(class_dir, f)
                        if self._is_valid_image(img_path):
                            all_images.append(img_path)

                # Sample random images from this class, limited by MAX_IMAGES_PER_CLASS
                if all_images:
                    sampled_images = random.sample(
                        all_images,
                        min(len(all_images), MAX_IMAGES_PER_CLASS)
                    )
                    self.class_to_images[cls] = sampled_images
                    print(f"  {cls}: {len(sampled_images)}/{len(all_images)} images sampled")

        # Update filtered classes and total number of samples
        self.classes = list(self.class_to_images.keys())
        print(f"Number of valid classes after sampling: {len(self.classes)}")
        print(f"Total sampled images: {sum(len(imgs) for imgs in self.class_to_images.values())}")

        self.all_pairs = self._generate_pairs()

    def _is_valid_image(self, img_path):
        """Checks if the file is a valid image."""
        try:
            with Image.open(img_path) as img:
                img.verify() # Checks the integrity of the image file
            return True
        except Exception as e:
            # print(f"Invalid image: {img_path} - {e}") # Can be enabled for debugging
            return False

    def _generate_pairs(self):
        """Creates positive and negative image pairs for training."""
        print("Generating pairs...")
        pairs = []

        for cls in self.classes:
            images = self.class_to_images[cls]
            if not images: continue # Skip if the image does not exist
            # print(f"Processing {cls} for pair generation...") # Can be enabled for debugging

            # Positive pairs (from the same class)
            positive_pairs = []
            if len(images) >= 2:
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        positive_pairs.append((images[i], images[j], 1)) # Label 1: Similar

            # Limit positive pairs
            if len(positive_pairs) > MAX_POSITIVE_PAIRS_PER_CLASS:
                positive_pairs = random.sample(positive_pairs, MAX_POSITIVE_PAIRS_PER_CLASS)
            pairs.extend(positive_pairs)

            # Negative pairs (from different classes)
            other_classes = [c for c in self.classes if c != cls]
            if other_classes:
                negative_pairs = []
                for i in range(len(images)):
                    # Create multiple negative matches for each positive image
                    for _ in range(min(3, len(other_classes))): # Maximum 3 negatives from different classes
                        neg_cls = random.choice(other_classes)
                        neg_image = random.choice(self.class_to_images[neg_cls]) # Random negative image
                        negative_pairs.append((images[i], neg_image, 0)) # Label 0: Different

                # Limit negative pairs
                if len(negative_pairs) > MAX_NEGATIVE_PAIRS_PER_CLASS:
                    negative_pairs = random.sample(negative_pairs, MAX_NEGATIVE_PAIRS_PER_CLASS)
                pairs.extend(negative_pairs)

        random.shuffle(pairs) # Shuffle pairs
        print(f"Total pairs generated: {len(pairs)}")
        return pairs

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        """Returns a pair and label from the dataset."""
        try:
            path1, path2, label = self.all_pairs[idx]

            # Load images safely
            img1 = self._load_image_safely(path1)
            img2 = self._load_image_safely(path2)

            if img1 is None or img2 is None:
                # If a corrupted or invalid image is found, select another random pair
                # We limit recursion to avoid infinite loops.
                # Here we simply return another pair.
                # A more robust approach would filter invalid pairs beforehand.
                print(f"Warning: Corrupted image detected at index {idx}. Retrying with another random pair.")
                return self.__getitem__(random.randint(0, len(self.all_pairs) - 1))

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, torch.tensor([label], dtype=torch.float32)

        except Exception as e:
            print(f"Error loading pair at index {idx}: {e}. Retrying with another random pair.")
            # Return another random pair in case of error
            return self.__getitem__(random.randint(0, len(self.all_pairs) - 1))

    def _load_image_safely(self, path):
        """Attempts to safely load the image."""
        try:
            img = Image.open(path).convert("RGB")
            # Minimum size check; very small images may cause problems
            if img.size[0] < 10 or img.size[1] < 10:
                print(f"Warning: Image too small ({img.size}) at {path}")
                return None
            return img
        except Exception as e:
            # print(f"Error loading image {path}: {e}") # Can be enabled for debugging
            return None


# === MODEL ===
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # ResNet-50 used for consistency with main classification model
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        # Final layer adjusted to match our embedding vector size
        base_model.fc = nn.Linear(base_model.fc.in_features, embedding_dim)
        self.feature_extractor = base_model

    def forward_once(self, x):
        """Runs one branch of the network."""
        return self.feature_extractor(x)

    def forward(self, x1, x2):
        """Passes two inputs through the network and returns embedding vectors."""
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


# === LOSS ===
class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        """Calculates contrastive loss."""
        dist = torch.nn.functional.pairwise_distance(out1, out2) # Euclidean distance
        # Loss: square distance for positives, square of margin-distance for negatives (if positive)
        loss = label * dist.pow(2) + (1 - label) * torch.clamp(self.margin - dist, min=0).pow(2)
        return loss.mean()


# === METRICS CALCULATION ===
def calculate_accuracy(out1, out2, labels, threshold=0.5):
    """Calculates accuracy and other metrics based on distance threshold."""
    distances = torch.nn.functional.pairwise_distance(out1, out2)
    predictions = (distances < threshold).float() # Similar if below threshold (1), else different (0)
    labels = labels.squeeze() # Make labels 1D

    # Convert to numpy for sklearn metrics
    pred_np = predictions.cpu().numpy()
    labels_np = labels.cpu().numpy()

    accuracy = accuracy_score(labels_np, pred_np)
    precision = precision_score(labels_np, pred_np, zero_division=0) # zero_division=0 to suppress warnings
    recall = recall_score(labels_np, pred_np, zero_division=0)
    f1 = f1_score(labels_np, pred_np, zero_division=0)

    return accuracy, precision, recall, f1


# === PLOTTING FUNCTIONS ===
def plot_training_history(train_losses, accuracies, precisions, recalls, f1_scores):
    """Visualizes and saves training history (loss and metrics)."""
    plt.figure(figsize=(15, 10))

    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('Training Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(accuracies, 'g-', linewidth=2)
    plt.title('Training Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)

    # Precision plot
    plt.subplot(2, 3, 3)
    plt.plot(precisions, 'r-', linewidth=2)
    plt.title('Training Precision', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)

    # Recall plot
    plt.subplot(2, 3, 4)
    plt.plot(recalls, 'orange', linewidth=2)
    plt.title('Training Recall', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)

    # F1 Score plot
    plt.subplot(2, 3, 5)
    plt.plot(f1_scores, 'purple', linewidth=2)
    plt.title('Training F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.grid(True, alpha=0.3)

    # Combined metrics plot
    plt.subplot(2, 3, 6)
    plt.plot(accuracies, 'g-', label='Accuracy', linewidth=2)
    plt.plot(precisions, 'r-', label='Precision', linewidth=2)
    plt.plot(recalls, 'orange', label='Recall', linewidth=2)
    plt.plot(f1_scores, 'purple', label='F1 Score', linewidth=2)
    plt.title('All Metrics Combined', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('siamese_training_history.png', dpi=300, bbox_inches='tight') # Updated filename
    plt.show()


# === TRAINING ===
def train_siamese(train_dir):
    print(f"Starting Siamese network training with directory: {train_dir}")

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print("Creating Siamese dataset...")
    dataset = FastSiameseDataset(train_dir, transform=transform)
    print(f"Dataset created with {len(dataset)} pairs for training.")

    if len(dataset) == 0:
        print("Error: No valid data pairs found for Siamese network training!")
        return

    # num_workers=0 (common for Windows multiprocessing issues)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training history lists
    train_losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    print("Siamese network training started...")
    start_time = time.time()

    model.train() # Set model to training mode

    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0.0
        batch_count = 0

        # Collect all outputs and labels for metric calculation
        all_out1, all_out2, all_labels = [], [], []

        try:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
            for img1, img2, label in progress_bar:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                optimizer.zero_grad() # Reset gradients
                out1, out2 = model(img1, img2) # Forward pass
                loss = criterion(out1, out2, label) # Calculate loss
                loss.backward() # Backpropagation
                optimizer.step() # Update parameters

                total_loss += loss.item()
                batch_count += 1

                # Append tensors to lists for metric calculation
                all_out1.append(out1.detach())
                all_out2.append(out2.detach())
                all_labels.append(label.detach())

                # Update progress bar
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        except Exception as e:
            print(f"Error during epoch {epoch + 1}: {e}")
            continue

        if batch_count > 0:
            # Calculate average loss
            avg_loss = total_loss / batch_count
            train_losses.append(avg_loss)

            # Calculate metrics
            all_out1_cat = torch.cat(all_out1, dim=0)
            all_out2_cat = torch.cat(all_out2, dim=0)
            all_labels_cat = torch.cat(all_labels, dim=0)

            accuracy, precision, recall, f1 = calculate_accuracy(all_out1_cat, all_out2_cat, all_labels_cat)

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

            epoch_time = time.time() - epoch_start_time

            print(f"Epoch {epoch + 1}/{epochs} - Time: {epoch_time:.2f}s")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print("-" * 50)

    total_time = time.time() - start_time
    print(f"\nSiamese network training completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epochs,
        'final_loss': train_losses[-1] if train_losses else 0, # "final_loss" is clearer than "loss"
        'final_accuracy': accuracies[-1] if accuracies else 0, # "final_accuracy"
        'training_history': {
            'losses': train_losses,
            'accuracies': accuracies,
            'precisions': precisions,
            'recalls': recalls,
            'f1_scores': f1_scores
        }
    }, "siamese_model_final.pth") # Updated filename

    print("Siamese model saved as siamese_model_final.pth")

    # Plot training history
    if train_losses:
        print("Creating Siamese training history plots...")
        plot_training_history(train_losses, accuracies, precisions, recalls, f1_scores)

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL SIAMESE TRAINING RESULTS")
    print("=" * 60)
    if train_losses:
        print(f"Final Loss: {train_losses[-1]:.4f}")
        print(f"Final Accuracy: {accuracies[-1]:.4f}")
        print(f"Final Precision: {precisions[-1]:.4f}")
        print(f"Final Recall: {recalls[-1]:.4f}")
        print(f"Final F1 Score: {f1_scores[-1]:.4f}")
        # Show best metrics as well
        print(f"Best Accuracy during training: {max(accuracies):.4f} (Epoch {accuracies.index(max(accuracies)) + 1})")
        print(f"Best F1 Score during training: {max(f1_scores):.4f} (Epoch {f1_scores.index(max(f1_scores)) + 1})")
    print("=" * 60)


if __name__ == "__main__":
    train_dir = r"C:\Users\anild\Desktop\Thesis\data1000\train" # Training data directory
    train_siamese(train_dir)
