import os
import shutil
import random

source_dir = r"/path/to/data/source"
train_dir = r"/path/to/data/train"
val_dir = r"/path/to/data/valid"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for group in os.listdir(source_dir):  # e.g., Aves, Mammalia, Serpentes
    group_path = os.path.join(source_dir, group)
    if not os.path.isdir(group_path):
        continue

    for species in os.listdir(group_path):  # e.g., Common Yellowthroat
        species_path = os.path.join(group_path, species)
        if not os.path.isdir(species_path):
            continue

        images = [f for f in os.listdir(species_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Training folder
        train_species_dir = os.path.join(train_dir, species)
        os.makedirs(train_species_dir, exist_ok=True)
        for img in train_images:
            shutil.copyfile(os.path.join(species_path, img),
                            os.path.join(train_species_dir, img))

        # Validation folder
        val_species_dir = os.path.join(val_dir, species)
        os.makedirs(val_species_dir, exist_ok=True)
        for img in val_images:
            shutil.copyfile(os.path.join(species_path, img),
                            os.path.join(val_species_dir, img))

print("Data successfully split and copied from subfolders.")
