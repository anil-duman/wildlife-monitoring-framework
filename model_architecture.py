import torch
from torch import nn
from torchvision import datasets, transforms, models
import numpy as np
import os


def get_data_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def print_model_architecture(model, num_classes):
    """Print detailed model architecture information"""
    print("=" * 80)
    print("MODEL ARCHITECTURE DETAILS")
    print("=" * 80)

    # 1. Full model architecture
    print("\n1. COMPLETE MODEL ARCHITECTURE:")
    print("-" * 50)
    print(model)

    # 2. Parameter counts
    print(f"\n2. PARAMETER INFORMATION:")
    print("-" * 50)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    # 3. Layer breakdown
    print(f"\n3. LAYER BREAKDOWN:")
    print("-" * 50)
    layer_types = {}
    for name, module in model.named_modules():
        layer_type = type(module).__name__
        if layer_type in layer_types:
            layer_types[layer_type] += 1
        else:
            layer_types[layer_type] = 1

    for layer_type, count in sorted(layer_types.items()):
        if count > 1:
            print(f"{layer_type}: {count} layers")
        else:
            print(f"{layer_type}: {count} layer")

    # 4. Key layer details
    print(f"\n4. KEY LAYER DETAILS:")
    print("-" * 50)

    # First conv layer
    first_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            first_conv = (name, module)
            break

    if first_conv:
        name, layer = first_conv
        print(f"First Conv Layer ({name}):")
        print(f"  Input channels: {layer.in_channels}")
        print(f"  Output channels: {layer.out_channels}")
        print(f"  Kernel size: {layer.kernel_size}")
        print(f"  Stride: {layer.stride}")
        print(f"  Padding: {layer.padding}")

    # Final classifier layer
    if hasattr(model, 'fc'):
        print(f"\nFinal Classifier (fc):")
        print(f"  Input features: {model.fc.in_features}")
        print(f"  Output classes: {model.fc.out_features}")
        print(f"  Layer type: {type(model.fc).__name__}")

    # 5. Model size estimation
    print(f"\n5. MODEL SIZE ESTIMATION:")
    print("-" * 50)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    print(f"Parameter size: {param_size / (1024 * 1024):.2f} MB")
    print(f"Buffer size: {buffer_size / (1024 * 1024):.2f} MB")
    print(f"Total estimated size: {model_size_mb:.2f} MB")

    # 6. Input/Output information
    print(f"\n6. INPUT/OUTPUT INFORMATION:")
    print("-" * 50)
    print(f"Expected input shape: (batch_size, 3, 224, 224)")
    print(f"Output shape: (batch_size, {num_classes})")
    print(f"Number of classes: {num_classes}")

    print("=" * 80)


def get_dataset_info(data_dir):
    """Get dataset information without loading all images"""
    if not os.path.exists(data_dir):
        print(f"Warning: Directory {data_dir} does not exist!")
        return None, 0

    try:
        # Create a minimal transform just to get class info
        minimal_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        dataset = datasets.ImageFolder(data_dir, transform=minimal_transform)
        return dataset.classes, len(dataset.classes)
    except Exception as e:
        print(f"Error loading dataset from {data_dir}: {e}")
        return None, 0


def view_model_architecture_only(train_dir, val_dir=None):
    """View model architecture without training"""
    print("VIEWING MODEL ARCHITECTURE (NO TRAINING)")
    print("=" * 60)

    # Get device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Get dataset information
    print(f"\nAnalyzing dataset from: {train_dir}")
    class_names, num_classes = get_dataset_info(train_dir)

    if class_names is None:
        print("Could not load dataset. Using default 10 classes for demonstration.")
        num_classes = 10
        class_names = [f"class_{i}" for i in range(num_classes)]
    else:
        print(f"Found {num_classes} classes: {class_names}")

    if val_dir and os.path.exists(val_dir):
        val_classes, val_num_classes = get_dataset_info(val_dir)
        if val_classes:
            print(f"Validation set has {val_num_classes} classes: {val_classes}")

    # Create model (same as in your training code)
    print(f"\nCreating ResNet-50 model for {num_classes} classes...")
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    # Modify final layer for your number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move to device
    model = model.to(device)

    # Print detailed architecture
    print_model_architecture(model, num_classes)

    # Save architecture to file
    save_architecture_to_file(model, num_classes, class_names)

    print("\nArchitecture analysis complete!")
    print("Architecture details saved to 'model_architecture.txt'")


def save_architecture_to_file(model, num_classes, class_names):
    """Save model architecture details to a text file"""
    with open('model_architecture.txt', 'w') as f:
        f.write("MODEL ARCHITECTURE REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("COMPLETE MODEL STRUCTURE:\n")
        f.write("-" * 30 + "\n")
        f.write(str(model) + "\n\n")

        # Parameter info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        f.write("PARAMETER INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n\n")

        # Class information
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Class names: {class_names}\n\n")

        # Model specifications
        f.write("MODEL SPECIFICATIONS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Base model: ResNet-50\n")
        f.write(f"Pre-trained: ImageNet weights\n")
        f.write(f"Input size: (3, 224, 224)\n")
        f.write(f"Output size: {num_classes} classes\n")


if __name__ == "__main__":
    # Update these paths according to your directory structure
    train_dir = r"/path/to/data/train"
    val_dir = r"/path/to/data/valid"

    # View model architecture without training
    view_model_architecture_only(train_dir, val_dir)