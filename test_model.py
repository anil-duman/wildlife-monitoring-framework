import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score, accuracy_score)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import time
from datetime import datetime
import os
import json
from tqdm import tqdm
from PIL import ImageFile
import random

# Allow corrupted images to be loaded (from training code)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed=42):
    """Seed function from training code"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data_transforms():
    """Transform function from training code"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def compute_metrics(y_true, y_pred):
    """Metrics function from training code"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Confusion matrix function from training code"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


class ModelTester:
    def __init__(self, test_dir, model_path, batch_size=32):
        # Set seed (from training code)
        set_seed()

        self.test_dir = test_dir
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use transform function from training code
        self.transform = get_data_transforms()

        # Create results directory
        self.results_dir = "test_results"
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"Using device: {self.device}")
        print(f"Results will be saved to: {self.results_dir}")

    def load_model_and_data(self):
        """Load model and test dataset"""
        print("Loading test dataset...")
        self.test_data = datasets.ImageFolder(self.test_dir, transform=self.transform)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size,
                                      shuffle=False, num_workers=4)  # num_workers added

        print(f"Test dataset: {len(self.test_data)} images, {len(self.test_data.classes)} classes")
        print(f"Classes: {self.test_data.classes}")

        print("Loading model...")
        # Use model structure from training code
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.test_data.classes))

        # Load model state
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print(f"Model loaded from: {self.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model.to(self.device)
        self.model.eval()

    def test_model(self):
        """Run comprehensive model testing with training-style validation"""
        print("\n" + "=" * 60)
        print("STARTING COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)

        start_time = time.time()
        all_preds = []
        all_labels = []
        all_probs = []
        running_loss = 0.0

        # Use criterion from training code
        criterion = nn.CrossEntropyLoss()

        print("Running inference on test set...")
        with torch.no_grad():
            # Use tqdm as in training code
            for inputs, labels in tqdm(self.test_loader, desc="Testing", leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)

                # Calculate loss (as in training code)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                # Get probabilities and predictions
                probs = torch.softmax(outputs, dim=1)
                predicted = torch.argmax(outputs, 1)  # As in training code

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        total_time = time.time() - start_time
        avg_loss = running_loss / len(self.test_loader)

        # Store results
        self.all_preds = np.array(all_preds)
        self.all_labels = np.array(all_labels)
        self.all_probs = np.array(all_probs)
        self.total_time = total_time
        self.avg_loss = avg_loss
        self.avg_sample_time = total_time / len(self.test_data)

        print(f"\nInference completed in {total_time:.2f} seconds")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Average time per sample: {self.avg_sample_time * 1000:.2f} ms")

    def calculate_metrics(self):
        """Calculate comprehensive metrics using training functions"""
        print("\nCalculating metrics...")

        # Use compute_metrics function from training code
        self.metrics = compute_metrics(self.all_labels, self.all_preds)

        # Confidence analysis
        self.confidence_scores = np.max(self.all_probs, axis=1)
        self.avg_confidence = np.mean(self.confidence_scores)
        self.correct_mask = (self.all_preds == self.all_labels)
        self.avg_confidence_correct = np.mean(self.confidence_scores[self.correct_mask])
        self.avg_confidence_incorrect = np.mean(self.confidence_scores[~self.correct_mask]) if np.sum(
            ~self.correct_mask) > 0 else 0

        # Per-class metrics (from sklearn)
        from sklearn.metrics import precision_recall_fscore_support
        self.precision_per_class, self.recall_per_class, self.f1_per_class, self.support_per_class = \
            precision_recall_fscore_support(self.all_labels, self.all_preds, zero_division=0)

    def print_results(self):
        """Print comprehensive results"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST RESULTS")
        print("=" * 60)

        print(f"Test Dataset: {len(self.test_data)} samples")
        print(f"Number of Classes: {len(self.test_data.classes)}")
        print(f"Inference Time: {self.total_time:.2f} seconds")
        print(f"Average Loss: {self.avg_loss:.4f}")
        print(f"Average Time per Sample: {self.avg_sample_time * 1000:.2f} ms")

        print(f"\n--- OVERALL METRICS (Training Style) ---")
        print(f"Accuracy: {self.metrics['accuracy']:.4f} ({self.metrics['accuracy'] * 100:.2f}%)")
        print(f"Precision (Weighted): {self.metrics['precision']:.4f}")
        print(f"Recall (Weighted): {self.metrics['recall']:.4f}")
        print(f"F1-Score (Weighted): {self.metrics['f1']:.4f}")

        print(f"\n--- CONFIDENCE ANALYSIS ---")
        print(f"Average Confidence: {self.avg_confidence:.4f}")
        print(f"Confidence (Correct Predictions): {self.avg_confidence_correct:.4f}")
        print(f"Confidence (Incorrect Predictions): {self.avg_confidence_incorrect:.4f}")

        # Per-class results
        print(f"\n--- PER-CLASS RESULTS ---")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 70)
        for i, class_name in enumerate(self.test_data.classes):
            print(f"{class_name:<20} {self.precision_per_class[i]:<10.4f} "
                  f"{self.recall_per_class[i]:<10.4f} {self.f1_per_class[i]:<10.4f} "
                  f"{self.support_per_class[i]:<10}")

    def save_results(self):
        """Save all results to files (training style)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Similar to save_metrics_to_csv in the training code
        test_results = {
            'Test_Results': {
                'accuracy': self.metrics['accuracy'],
                'precision': self.metrics['precision'],
                'recall': self.metrics['recall'],
                'f1': self.metrics['f1'],
                'avg_loss': self.avg_loss,
                'inference_time': self.total_time,
                'avg_confidence': self.avg_confidence
            }
        }

        # Save in CSV format
        df_results = pd.DataFrame([test_results['Test_Results']], index=['Test_Results'])
        df_results.to_csv(f'{self.results_dir}/test_results_{timestamp}.csv')

        # Per-class results
        per_class_results = []
        for i, class_name in enumerate(self.test_data.classes):
            per_class_results.append({
                'class_name': class_name,
                'precision': self.precision_per_class[i],
                'recall': self.recall_per_class[i],
                'f1_score': self.f1_per_class[i],
                'support': int(self.support_per_class[i])
            })

        df_per_class = pd.DataFrame(per_class_results)
        df_per_class.to_csv(f'{self.results_dir}/per_class_metrics_{timestamp}.csv', index=False)

        # Detailed predictions
        predictions_df = pd.DataFrame({
            'true_label': [self.test_data.classes[label] for label in self.all_labels],
            'predicted_label': [self.test_data.classes[pred] for pred in self.all_preds],
            'confidence': self.confidence_scores,
            'correct': self.correct_mask
        })
        predictions_df.to_csv(f'{self.results_dir}/predictions_{timestamp}.csv', index=False)

        print(f"\n--- RESULTS SAVED ---")
        print(f"Test results: test_results_{timestamp}.csv")
        print(f"Per-class metrics: per_class_metrics_{timestamp}.csv")
        print(f"All predictions: predictions_{timestamp}.csv")

    def plot_visualizations(self):
        """Create visualizations using training functions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use the plot_confusion_matrix function in the training code
        plot_confusion_matrix(self.all_labels, self.all_preds, self.test_data.classes,
                              'Test Set Confusion Matrix',
                              f'{self.results_dir}/confusion_matrix_{timestamp}.png')

        # Confidence distribution and F1 scores
        plt.figure(figsize=(15, 5))

        # Confidence distribution
        plt.subplot(1, 3, 1)
        plt.hist(self.confidence_scores[self.correct_mask], bins=30, alpha=0.7,
                 label='Correct', color='green', density=True)
        plt.hist(self.confidence_scores[~self.correct_mask], bins=30, alpha=0.7,
                 label='Incorrect', color='red', density=True)
        plt.xlabel('Confidence Score')
        plt.ylabel('Density')
        plt.title('Confidence Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Per-class F1 scores
        plt.subplot(1, 3, 2)
        plt.barh(range(len(self.test_data.classes)), self.f1_per_class)
        plt.yticks(range(len(self.test_data.classes)), self.test_data.classes)
        plt.xlabel('F1 Score')
        plt.title('Per-Class F1 Scores')
        plt.grid(True, alpha=0.3)

        # Accuracy by confidence bins
        plt.subplot(1, 3, 3)
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []

        for i in range(len(bins) - 1):
            mask = (self.confidence_scores >= bins[i]) & (self.confidence_scores < bins[i + 1])
            if np.sum(mask) > 0:
                acc = np.mean(self.correct_mask[mask])
                bin_accuracies.append(acc)
            else:
                bin_accuracies.append(0)

        plt.plot(bin_centers, bin_accuracies, 'o-')
        plt.xlabel('Confidence Bin')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Confidence')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/analysis_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_classification_report(self):
        """Generate and save detailed classification report (training style)"""
        print(f"\n=== Detailed Classification Report (Test) ===")
        report = classification_report(self.all_labels, self.all_preds,
                                       target_names=self.test_data.classes)
        print(report)

        # Save classification report (as in training code)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'{self.results_dir}/classification_report_{timestamp}.txt', 'w') as f:
            f.write("Test Set Classification Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Test Loss: {self.avg_loss:.4f}\n")
            f.write(f"Test Accuracy: {self.metrics['accuracy']:.4f}\n")
            f.write(f"Test F1 Score: {self.metrics['f1']:.4f}\n\n")
            f.write(report)

    def run_complete_evaluation(self):
        """Run complete evaluation pipeline (training style)"""
        self.load_model_and_data()
        self.test_model()
        self.calculate_metrics()
        self.print_results()
        self.save_results()
        self.plot_visualizations()
        self.generate_classification_report()

        print(f"\n=== Test Complete ===")
        print(f"Test Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Test F1 Score: {self.metrics['f1']:.4f}")
        print("All results and visualizations saved!")


if __name__ == "__main__":

    test_dir = r"C:\Users\anild\Desktop\Thesis\data1000\test"
    model_path = r"C:\Users\anild\PycharmProjects\Species_Classification\best_model_kfold.pth"  # Düzeltildi
    batch_size = 32

    # Run comprehensive testing
    tester = ModelTester(test_dir, model_path, batch_size)
    tester.run_complete_evaluation()

    print(f"\n{'=' * 60}")
    print("EVALUATION COMPLETE!")
    print(f"{'=' * 60}")