"""
Quick script to check if model files exist and their sizes
Run this on Hugging Face to verify LFS files downloaded
"""
import os
from pathlib import Path

print("=" * 60)
print("MODEL FILES CHECK")
print("=" * 60)

# Check current working directory
print(f"\nCurrent directory: {os.getcwd()}")
print(f"Files in current dir: {os.listdir('.')[:10]}")

# Check if saved_models folder exists
if os.path.exists("saved_models"):
    print(f"\n✅ saved_models folder EXISTS")
    print(f"Files in saved_models: {os.listdir('saved_models')}")
    
    # Check each model file
    models = {
        "lung_disease_model.h5": 286_932_184,
        "breast_cancer_image_model_improved.h5": 164_625_840,
        "diabetes_model_optimized.joblib": 2_652_777,
        "kidney_model.joblib": 56_793
    }
    
    print("\nMODEL FILE SIZES:")
    print("-" * 60)
    for model_name, expected_size in models.items():
        model_path = f"saved_models/{model_name}"
        if os.path.exists(model_path):
            actual_size = os.path.getsize(model_path)
            if actual_size < 1000:  # LFS pointer file is ~130 bytes
                print(f"❌ {model_name}")
                print(f"   Size: {actual_size} bytes (LFS POINTER - NOT DOWNLOADED!)")
                print(f"   Expected: {expected_size:,} bytes")
            elif abs(actual_size - expected_size) < 1000:
                print(f"✅ {model_name}")
                print(f"   Size: {actual_size:,} bytes (CORRECT)")
            else:
                print(f"⚠️  {model_name}")
                print(f"   Size: {actual_size:,} bytes")
                print(f"   Expected: {expected_size:,} bytes")
        else:
            print(f"❌ {model_name} - FILE NOT FOUND")
    
else:
    print("\n❌ saved_models folder DOES NOT EXIST")
    print("Available directories:", [d for d in os.listdir('.') if os.path.isdir(d)])

print("\n" + "=" * 60)

