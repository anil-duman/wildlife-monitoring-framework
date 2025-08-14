import os
import shutil
import random

# === DIRECTORIES ===
source_dir = r"/path/to/data/source"
target_dir = r"/path/to/data/target"

max_images = 100  # Maximum 100 images per species

os.makedirs(target_dir, exist_ok=True)

for species in os.listdir(source_dir):
    species_path = os.path.join(source_dir, species)
    if not os.path.isdir(species_path):
        continue  # Skip if not a folder

    images = [f for f in os.listdir(species_path) if os.path.isfile(os.path.join(species_path, f))]

    if len(images) == 0:
        print(f"Warning: No images found in '{species}' folder, skipping.")
        continue

    selected_images = random.sample(images, min(len(images), max_images))

    target_species_path = os.path.join(target_dir, species)
    os.makedirs(target_species_path, exist_ok=True)

    for image_file in selected_images:
        src = os.path.join(species_path, image_file)
        dst = os.path.join(target_species_path, image_file)
        shutil.copyfile(src, dst)

    print(f"{species}: {len(selected_images)} images copied.")
