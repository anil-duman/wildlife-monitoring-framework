import os
import shutil
import random

source_dir = r"/path/to/source"
target_dir = r"/path/to/target"

max_images = 1000

os.makedirs(target_dir, exist_ok=True)

for species in os.listdir(source_dir):
    species_path = os.path.join(source_dir, species)
    if not os.path.isdir(species_path):
        continue  # Skip if not a directory

    images = [f for f in os.listdir(species_path) if os.path.isfile(os.path.join(species_path, f))]
    selected_images = random.sample(images, min(len(images), max_images))

    target_species_path = os.path.join(target_dir, species)
    os.makedirs(target_species_path, exist_ok=True)

    for image_file in selected_images:
        src = os.path.join(species_path, image_file)
        dst = os.path.join(target_species_path, image_file)
        shutil.copyfile(src, dst)
