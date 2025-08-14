import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, \
    classification_report
import numpy as np
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageFile
import pandas as pd

# Allow corrupted images to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_data_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), all_preds, all_labels


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), all_preds, all_labels


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
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


def save_metrics_to_csv(all_fold_metrics, avg_metrics, final_test_metrics=None):
    # K-fold results
    df_folds = pd.DataFrame(all_fold_metrics)
    df_folds.index = [f'Fold_{i + 1}' for i in range(len(all_fold_metrics))]

    # Add average row
    df_avg = pd.DataFrame([avg_metrics], index=['Average'])
    df_results = pd.concat([df_folds, df_avg])

    # Add final test if available
    if final_test_metrics:
        df_final = pd.DataFrame([final_test_metrics], index=['Final_Test'])
        df_results = pd.concat([df_results, df_final])

    df_results.to_csv('training_results.csv')
    print("Results saved to training_results.csv")


def cross_validate_and_test(train_dir, val_dir, k_folds=5, batch_size=32, epochs=10):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = get_data_transforms()

    # Load training dataset for K-fold
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    print(f"Training dataset: {len(train_dataset)} images, {len(train_dataset.classes)} classes")

    # Load validation dataset for final test
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    print(f"Validation dataset: {len(val_dataset)} images")

    # K-fold cross validation on training data
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    targets = [sample[1] for sample in train_dataset.samples]
    all_fold_metrics = []

    best_fold_model = None
    best_fold_score = 0.0

    print(f"\n=== Starting {k_folds}-Fold Cross Validation ===")

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets)):
        print(f"\n--- Fold {fold + 1}/{k_folds} ---")
        print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")

        # Create subsets
        fold_train_subset = Subset(train_dataset, train_idx)
        fold_val_subset = Subset(train_dataset, val_idx)

        train_loader = DataLoader(fold_train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(fold_val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Initialize model
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

        best_val_f1 = 0.0

        # Training loop
        for epoch in range(epochs):
            train_loss, train_preds, train_labels = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_labels = validate(model, val_loader, criterion, device)

            train_metrics = compute_metrics(train_labels, train_preds)
            val_metrics = compute_metrics(val_labels, val_preds)

            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{epochs}:")
            print(
                f"  Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")

            # Save best model for this fold
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                if val_metrics['f1'] > best_fold_score:
                    best_fold_score = val_metrics['f1']
                    best_fold_model = model.state_dict().copy()
                    torch.save(best_fold_model, f'best_model_fold_{fold + 1}.pth')

        # Final validation metrics for this fold
        final_val_loss, final_val_preds, final_val_labels = validate(model, val_loader, criterion, device)
        fold_metrics = compute_metrics(final_val_labels, final_val_preds)
        all_fold_metrics.append(fold_metrics)

        print(f"Fold {fold + 1} Final Results:")
        for metric, value in fold_metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")

    # Calculate average metrics across folds
    print(f"\n=== K-Fold Cross Validation Results ===")
    avg_metrics = {key: np.mean([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}
    std_metrics = {key: np.std([m[key] for m in all_fold_metrics]) for key in all_fold_metrics[0]}

    for metric in avg_metrics:
        print(f"{metric.capitalize()}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

    # Save best model
    torch.save(best_fold_model, 'best_model_kfold.pth')
    print(f"\nBest model saved with F1 score: {best_fold_score:.4f}")

    # Final test on validation set
    print(f"\n=== Final Test on Validation Set ===")

    # Load best model
    final_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    final_model.fc = nn.Linear(final_model.fc.in_features, len(train_dataset.classes))
    final_model.load_state_dict(best_fold_model)
    final_model = final_model.to(device)

    # Test on validation set
    val_test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loss, test_preds, test_labels = validate(final_model, val_test_loader, criterion, device)
    final_test_metrics = compute_metrics(test_labels, test_preds)

    print("Final Test Results:")
    for metric, value in final_test_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

    # Generate detailed classification report
    print(f"\n=== Detailed Classification Report (Final Test) ===")
    report = classification_report(test_labels, test_preds, target_names=train_dataset.classes)
    print(report)

    # Save classification report
    with open('classification_report.txt', 'w') as f:
        f.write("Final Test Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(report)

    # Plot confusion matrices
    plot_confusion_matrix(test_labels, test_preds, train_dataset.classes,
                          'Final Test Confusion Matrix', 'confusion_matrix_final_test.png')

    # Save all results to CSV
    save_metrics_to_csv(all_fold_metrics, avg_metrics, final_test_metrics)

    print(f"\n=== Training Complete ===")
    print(f"Best K-fold F1 Score: {best_fold_score:.4f}")
    print(f"Final Test F1 Score: {final_test_metrics['f1']:.4f}")
    print("All results and visualizations saved!")


if __name__ == "__main__":
    # Update these paths according to your directory structure
    train_dir = r"/path/to/data/train"
    val_dir = r"/path/to/data/valid"

    cross_validate_and_test(
        train_dir=train_dir,
        val_dir=val_dir,
        k_folds=5,
        batch_size=32,
        epochs=10
    )