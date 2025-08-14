import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.ensemble import IsolationForest
import numpy as np

# === CONFIGURATION ===
metadata_path = r'/path/to/metadata'  # Metadata path
filtered_images_folder = r'/path/to/filtered_images'  # Main folder of filtered images


# === Function: List all files in subfolders ===
def list_all_files(folder):
    """Lists the names of all files in the specified folder and subfolders."""
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            all_files.append(file)
    return all_files


# === LOAD METADATA ===
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

with open(metadata_path, 'r') as f:
    metadata = json.load(f)

image_data = metadata['images']
annotation_data = metadata.get('annotations', [])

# === GET FILE NAMES IN SUB-FOLDERS ===
used_filenames = set(list_all_files(filtered_images_folder))
print(f"Filtered images folder contains {len(used_filenames)} files.")

# === FILTER METADATA: Only take those matching the filtered files ===
filtered_metadata = [img for img in image_data if os.path.basename(img['file_name']) in used_filenames]

if len(filtered_metadata) == 0:
    raise ValueError("Filtered images do not match the metadata. Check file names and paths.")

df = pd.DataFrame(filtered_metadata)

# === DATETIME PARSING ===
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date
df['month'] = df['datetime'].dt.month
df['year'] = df['datetime'].dt.year


# === SPECIES EXTRACTION ===
def extract_species_improved(file_path):
    """Improved species extraction function"""
    parts = file_path.replace('\\', '/').split('/')

    # Try different scenarios:
    # Scenario 1: images/group/species/image.jpg format
    if len(parts) >= 4:
        potential_species = parts[-2]  # The last folder is usually the species
        if potential_species and potential_species not in ['images', 'train', 'val', 'test']:
            return potential_species

    # Scenario 2: More general approach - the last folder
    if len(parts) >= 3:
        potential_species = parts[-2]
        if potential_species and potential_species != 'images':
            return potential_species

    # Scenario 3: Extract species from the file name
    filename = parts[-1]
    if '_' in filename:
        potential_species = filename.split('_')[0]
        if potential_species and not potential_species.isdigit():
            return potential_species

    return 'Unknown'


df['species'] = df['file_name'].apply(extract_species_improved)

# === DEBUG: Species extraction analysis ===
print("=== DEBUG: Species extraction analysis ===")
print(f"Total filtered metadata entries: {len(df)}")
print(f"Sample file_name entries (first 10):")
for i in range(min(10, len(df))):
    print(f"  {i}: {df.iloc[i]['file_name']}")

current_species = df['species'].value_counts()
print(f"\nCurrent species extraction results (first 10):")
print(current_species.head(10))

# === ANNOTATION PROCESSING ===
print("Processing annotations...")
annotation_map = {ann['image_id']: ann['category_id'] for ann in annotation_data}


def label_presence(file_id):
    return 'Blank' if annotation_map.get(file_id, -1) == 0 else 'Animal'


df['presence'] = df['id'].apply(label_presence)

# === ANALYSIS STARTS HERE ===
print("\n=== Creating Analysis Charts ===")

# Animal presence check
animal_df = df[df['presence'] == 'Animal'] if 'presence' in df.columns else df
species_counts = animal_df['species'].value_counts()
location_counts = df['location'].value_counts()

# === 1. COMBINED SPECIES ANALYSIS (Top 15 + Bottom 10) ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

# Top 15 Species
top_species = species_counts.head(15)
bars1 = ax1.bar(range(len(top_species)), top_species.values, color='skyblue', alpha=0.8, edgecolor='navy')
ax1.set_title('Top 15 Most Detected Species', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Species', fontsize=12)
ax1.set_ylabel('Detection Count', fontsize=12)
ax1.set_xticks(range(len(top_species)))
ax1.set_xticklabels(top_species.index, rotation=45, ha='right', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

# Display values on the bar
for i, v in enumerate(top_species.values):
    ax1.text(i, v + max(top_species.values) * 0.01, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Bottom 10 Rare Species
bottom_species = species_counts.tail(10)
bottom_species = bottom_species[bottom_species > 0]  # Those greater than 0

bars2 = ax2.bar(range(len(bottom_species)), bottom_species.values, color='lightcoral', alpha=0.8, edgecolor='darkred')
ax2.set_title('Bottom 10 Least Detected Species (Rare)', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Species', fontsize=12)
ax2.set_ylabel('Detection Count', fontsize=12)
ax2.set_xticks(range(len(bottom_species)))
ax2.set_xticklabels(bottom_species.index, rotation=45, ha='right', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

for i, v in enumerate(bottom_species.values):
    ax2.text(i, v + max(bottom_species.values) * 0.05, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('01_species_analysis_combined.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# === 2. COMBINED LOCATION ANALYSIS (Top 15 + Bottom 10) ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

# Top 15 Locations
top_locations = location_counts.head(15)
bars1 = ax1.bar(range(len(top_locations)), top_locations.values, color='lightgreen', alpha=0.8, edgecolor='darkgreen')
ax1.set_title('Top 15 Most Active Camera Locations', fontsize=16, fontweight='bold', pad=20)
ax1.set_xlabel('Location ID', fontsize=12)
ax1.set_ylabel('Detection Count', fontsize=12)
ax1.set_xticks(range(len(top_locations)))
ax1.set_xticklabels(top_locations.index, rotation=45, ha='right', fontsize=10)
ax1.grid(axis='y', alpha=0.3)

for i, v in enumerate(top_locations.values):
    ax1.text(i, v + max(top_locations.values) * 0.01, str(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

# Bottom 10 Locations
bottom_locations = location_counts.tail(10)
bars2 = ax2.bar(range(len(bottom_locations)), bottom_locations.values, color='khaki', alpha=0.8, edgecolor='olive')
ax2.set_title('Bottom 10 Least Active Camera Locations', fontsize=16, fontweight='bold', pad=20)
ax2.set_xlabel('Location ID', fontsize=12)
ax2.set_ylabel('Detection Count', fontsize=12)
ax2.set_xticks(range(len(bottom_locations)))
ax2.set_xticklabels(bottom_locations.index, rotation=45, ha='right', fontsize=10)
ax2.grid(axis='y', alpha=0.3)

for i, v in enumerate(bottom_locations.values):
    ax2.text(i, v + max(bottom_locations.values) * 0.05, str(v), ha='center', va='bottom', fontsize=9,
             fontweight='bold')

plt.tight_layout()
plt.savefig('02_location_analysis_combined.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# === 3. SPECIES DIVERSITY BY LOCATION ===
plt.figure(figsize=(16, 8))
location_species_diversity = animal_df.groupby('location')['species'].nunique().sort_values(ascending=False)
top_diverse_locations = location_species_diversity.head(20)

bars = plt.bar(range(len(top_diverse_locations)), top_diverse_locations.values, color='purple', alpha=0.7,
               edgecolor='darkviolet')
plt.title('Top 20 Locations by Species Diversity', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Location ID', fontsize=12)
plt.ylabel('Number of Different Species', fontsize=12)
plt.xticks(range(len(top_diverse_locations)), top_diverse_locations.index, rotation=45, ha='right', fontsize=10)
plt.grid(axis='y', alpha=0.3)

for i, v in enumerate(top_diverse_locations.values):
    plt.text(i, v + max(top_diverse_locations.values) * 0.01, str(v), ha='center', va='bottom', fontsize=9,
             fontweight='bold')

plt.tight_layout()
plt.savefig('03_species_diversity_by_location.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# === 4. DIURNAL ACTIVITY ===
plt.figure(figsize=(14, 8))
hours, counts = np.unique(df['hour'], return_counts=True)
bars = plt.bar(hours, counts, color='orange', alpha=0.7, edgecolor='darkorange')
plt.title('Diurnal Activity Pattern (24-hour)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Number of Detections', fontsize=12)
plt.xticks(range(0, 24, 2))
plt.grid(axis='y', alpha=0.3)

# Highlight the peak hours
max_hour = hours[np.argmax(counts)]
plt.axvline(x=max_hour, color='red', linestyle='--', alpha=0.7, label=f'Peak Hour: {max_hour}:00')
plt.legend()

# Display values on the bar (only for high ones)
for i, v in zip(hours, counts):
    if v > np.percentile(counts, 75):  # Show for the top 25%
        plt.text(i, v + max(counts) * 0.01, str(v), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('04_diurnal_activity.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# === 5. SEASONAL ACTIVITY ===
plt.figure(figsize=(14, 8))
months, counts = np.unique(df['month'], return_counts=True)
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
bars = plt.bar(months, counts, color='green', alpha=0.7, edgecolor='darkgreen')
plt.title('Seasonal Activity Pattern (Monthly)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Detections', fontsize=12)
plt.xticks(months, [month_names[m - 1] for m in months], rotation=45)
plt.grid(axis='y', alpha=0.3)

# Highlight the peak month
max_month = months[np.argmax(counts)]
plt.axvline(x=max_month, color='red', linestyle='--', alpha=0.7, label=f'Peak Month: {month_names[max_month - 1]}')
plt.legend()

for i, v in zip(months, counts):
    plt.text(i, v + max(counts) * 0.01, str(v), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('05_seasonal_activity.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()

# === 6. ANOMALY DETECTION FOR RARE SPECIES ===
print("Starting anomaly detection for rare species...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Used as a ResNet-50 feature extractor
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove the final classification layer
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def find_image_file(filename_in_metadata, base_folder):
    """Finds the real file path using the file name from metadata."""
    target_filename = os.path.basename(filename_in_metadata)
    for root, dirs, files in os.walk(base_folder):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None


embeddings = []
file_paths = []
max_images_for_anomaly_detection = 1000

print(f"Extracting embeddings for anomaly detection (max {max_images_for_anomaly_detection} images)...")

processed_count = 0
for i, row in df.iterrows():
    if processed_count >= max_images_for_anomaly_detection:
        break

    filename = os.path.basename(row['file_name'])
    img_path = find_image_file(filename, filtered_images_folder)

    if img_path and os.path.exists(img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = resnet(tensor).squeeze().cpu().numpy()
                embeddings.append(emb)
                file_paths.append(img_path)
                processed_count += 1

            if processed_count % 100 == 0:
                print(f"Processed {processed_count} images for embedding extraction...")

        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            continue

print(f"Successfully extracted embeddings from {len(embeddings)} images for anomaly detection.")

# Anomaly detection
if len(embeddings) >= 10:
    print("Running Isolation Forest for anomaly detection...")
    X = np.array(embeddings)
    print(f"Embeddings shape: {X.shape}")

    contamination_rate = min(0.1, max(0.01, 20.0 / len(embeddings)))
    print(f"Using contamination rate: {contamination_rate:.4f}")

    iso_forest = IsolationForest(contamination=contamination_rate, random_state=42)
    iso_forest.fit(X)

    anomaly_scores = iso_forest.decision_function(X)
    predictions = iso_forest.predict(X)
    anomaly_indices = np.where(predictions == -1)[0]

    print(f"Found {len(anomaly_indices)} potential anomalies.")

    sorted_indices = np.argsort(anomaly_scores)[:len(anomaly_indices)]
    os.makedirs('rare_species_candidates', exist_ok=True)

    saved_anomaly_count = 0
    num_anomalies_to_save = min(20, len(sorted_indices))

    print(f"Saving top {num_anomalies_to_save} rare species candidates...")

    for rank, idx in enumerate(sorted_indices[:num_anomalies_to_save]):
        try:
            if os.path.exists(file_paths[idx]):
                img = Image.open(file_paths[idx])
                save_name = f'rare_species_candidates/rare_candidate_{rank + 1}_{os.path.basename(file_paths[idx])}'
                img.save(save_name)
                print(
                    f"Saved candidate {rank + 1}/{num_anomalies_to_save}: {os.path.basename(file_paths[idx])} (Score: {anomaly_scores[idx]:.4f})")
                saved_anomaly_count += 1
        except Exception as e:
            print(f"Error saving anomaly candidate {file_paths[idx]}: {e}")

    print(f"Total {saved_anomaly_count} rare species candidate images saved in 'rare_species_candidates/' folder.")

# === SUMMARY STATISTICS ===
print("\n" + "=" * 60)
print("FINAL ANALYSIS SUMMARY")
print("=" * 60)
print(f"Total Images Analyzed: {len(df):,}")
print(f"Total Unique Species: {len(species_counts)}")
print(f"Total Camera Locations: {len(location_counts)}")
print(f"Analysis Period: {df['datetime'].min().strftime('%Y-%m-%d')} to {df['datetime'].max().strftime('%Y-%m-%d')}")
print()
print("TOP 5 SPECIES:")
for i, (species, count) in enumerate(species_counts.head(5).items()):
    print(f"  {i + 1}. {species}: {count:,} detections")
print()
print("TOP 5 LOCATIONS:")
for i, (location, count) in enumerate(location_counts.head(5).items()):
    print(f"  {i + 1}. {location}: {count:,} detections")
print()
print("RARE SPECIES (≤5 detections):")
rare_species = species_counts[species_counts <= 5]
if len(rare_species) > 0:
    for species, count in rare_species.items():
        print(f"  • {species}: {count} detections")
    print(f"  Total: {len(rare_species)} rare species")
else:
    print("  • No rare species found (all species have >5 detections)")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETED!")
print("Created PDF files:")
print("  • 01_species_analysis_combined.pdf (Top 15 + Bottom 10 species)")
print("  • 02_location_analysis_combined.pdf (Top 15 + Bottom 10 locations)")
print("  • 03_species_diversity_by_location.pdf")
print("  • 04_diurnal_activity.pdf")
print("  • 05_seasonal_activity.pdf")
print("=" * 60)

print("All visualizations and analysis completed.")