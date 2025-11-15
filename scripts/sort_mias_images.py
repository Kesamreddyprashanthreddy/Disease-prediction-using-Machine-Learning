# File: sort_mias_images.py (Final version with manual parsing)

import os
import shutil

print("--- Starting MIAS Image Sorting Script ---")

# --- ⚙️ CONFIGURATION ⚙️ ---
SOURCE_IMAGES_DIR = "all-mias" 
LABELS_FILE_NAME = "Info.txt"
SORTED_IMAGES_DIR = "mias_images_sorted"
# ------------------------------------

# --- Main Logic ---

# Create the destination folders
benign_dir = os.path.join(SORTED_IMAGES_DIR, "Benign")
malignant_dir = os.path.join(SORTED_IMAGES_DIR, "Malignant")
os.makedirs(benign_dir, exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)
print(f"Destination folders are ready.")

benign_count = 0
malignant_count = 0

try:
    with open(LABELS_FILE_NAME, 'r') as f:
        print(f"Successfully loaded {LABELS_FILE_NAME}. Parsing manually...")
        
        for line in f:
            # Split the line into a list of words
            words = line.split()
            
            # Skip any empty lines
            if not words:
                continue

            # The most reliable way to find relevant rows is to check for 'B' or 'M'
            severity = None
            if 'B' in words:
                severity = 'B'
            elif 'M' in words:
                severity = 'M'

            # If we found a Benign or Malignant case, process it
            if severity:
                refnum = words[0]
                image_filename = refnum + ".pgm"
                source_path = os.path.join(SOURCE_IMAGES_DIR, image_filename)

                if os.path.exists(source_path):
                    if severity == 'B':
                        shutil.copy(source_path, os.path.join(benign_dir, image_filename))
                        benign_count += 1
                    elif severity == 'M':
                        shutil.copy(source_path, os.path.join(malignant_dir, image_filename))
                        malignant_count += 1
                else:
                    print(f"Warning: Image {image_filename} not found in '{SOURCE_IMAGES_DIR}'.")

except FileNotFoundError:
    print(f"❌ Error: Could not find '{LABELS_FILE_NAME}'.")
    exit()

print("\n--- Sorting Complete ---")
print(f"✅ Sorted {benign_count} images into the 'Benign' folder.")
print(f"✅ Sorted {malignant_count} images into the 'Malignant' folder.")