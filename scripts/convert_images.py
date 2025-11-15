# File: convert_images.py

import os
from PIL import Image

print("--- Starting Image Conversion Script ---")

# --- Configuration ---
# The folder where your sorted .pgm files are
SOURCE_DIR = "mias_images_sorted" 
# The new folder where the .png files will be saved
DESTINATION_DIR = "mias_images_png"

# --- Main Logic ---

# Check if the source directory exists
if not os.path.exists(SOURCE_DIR):
    print(f"❌ Error: Source directory '{SOURCE_DIR}' not found. Please run the sorting script first.")
    exit()

converted_count = 0
# Loop through the subdirectories (Benign, Malignant)
for class_name in os.listdir(SOURCE_DIR):
    class_dir_source = os.path.join(SOURCE_DIR, class_name)
    class_dir_dest = os.path.join(DESTINATION_DIR, class_name)
    
    # Create the new destination subdirectories
    os.makedirs(class_dir_dest, exist_ok=True)
    
    # Find all .pgm files and convert them
    if os.path.isdir(class_dir_source):
        print(f"Converting images in '{class_dir_source}'...")
        for filename in os.listdir(class_dir_source):
            if filename.endswith(".pgm"):
                # Open the .pgm image
                img_path_source = os.path.join(class_dir_source, filename)
                img = Image.open(img_path_source)
                
                # Create the new .png filename and save
                new_filename = os.path.splitext(filename)[0] + ".png"
                img_path_dest = os.path.join(class_dir_dest, new_filename)
                img.save(img_path_dest)
                converted_count += 1

print("\n--- Conversion Complete ---")
print(f"✅ Successfully converted {converted_count} images to PNG format.")
print(f"Your new dataset is ready in the '{DESTINATION_DIR}' folder.")