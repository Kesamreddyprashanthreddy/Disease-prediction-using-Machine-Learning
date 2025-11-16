"""
Model Download Helper
Downloads large model files from cloud storage if not present locally
"""
import os
import requests
import streamlit as st
from pathlib import Path

def download_file_from_url(url, destination, file_name):
    """Download a file from URL with progress bar."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    if os.path.exists(destination):
        file_size = os.path.getsize(destination)
        if file_size > 1024:  # More than 1KB (not LFS pointer)
            return True
    
    try:
        st.info(f"⬇️ Downloading {file_name}... This may take a few minutes.")
        
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                progress_bar = st.progress(0)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = int((downloaded / total_size) * 100)
                        progress_bar.progress(progress)
                progress_bar.empty()
        
        st.success(f"✅ Downloaded {file_name}")
        return True
    except Exception as e:
        st.error(f"❌ Failed to download {file_name}: {str(e)}")
        return False

def ensure_models_downloaded():
    """
    Ensure all required models are downloaded.
    If deploying to Streamlit Cloud, upload your models to:
    - GitHub Releases
    - Google Drive (with gdown)
    - Hugging Face Hub
    - Dropbox, AWS S3, etc.
    
    Then update the URLs below.
    """
    
    models_dir = Path("saved_models")
    models_dir.mkdir(exist_ok=True)
    
    # OPTION 1: Using GitHub Releases (RECOMMENDED)
    # Upload your models to GitHub Releases, then use these URLs:
    # Example: https://github.com/USERNAME/REPO/releases/download/v1.0/lung_disease_model.h5
    
    # OPTION 2: Using Hugging Face Hub (BEST FOR ML MODELS)
    # Upload to: https://huggingface.co/YOUR_USERNAME/YOUR_MODEL
    # Then use: https://huggingface.co/YOUR_USERNAME/YOUR_MODEL/resolve/main/lung_disease_model.h5
    
    # OPTION 3: Using Google Drive (with public link)
    # Share file → Anyone with link → Copy link
    # Convert to download URL using gdown
    
    models_config = {
        # Format: "local_path": "download_url"
        # UPDATE THESE URLs WITH YOUR ACTUAL MODEL LOCATIONS
        
        # Example using GitHub Releases:
        # "saved_models/lung_disease_model.h5": 
        #     "https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning/releases/download/v1.0/lung_disease_model.h5",
        
        # Example using Hugging Face:
        # "saved_models/lung_disease_model.h5":
        #     "https://huggingface.co/YOUR_USERNAME/disease-models/resolve/main/lung_disease_model.h5",
    }
    
    # Try to download missing models
    for model_path, url in models_config.items():
        if url and not os.path.exists(model_path):
            download_file_from_url(url, model_path, os.path.basename(model_path))
    
    return True

# Alternative: Download from Google Drive using gdown
def download_from_google_drive(file_id, destination, file_name):
    """Download from Google Drive using gdown."""
    try:
        import gdown
        
        if os.path.exists(destination) and os.path.getsize(destination) > 1024:
            return True
        
        st.info(f"⬇️ Downloading {file_name} from Google Drive...")
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, destination, quiet=False)
        st.success(f"✅ Downloaded {file_name}")
        return True
    except ImportError:
        st.error("❌ Please install gdown: pip install gdown")
        return False
    except Exception as e:
        st.error(f"❌ Failed to download from Google Drive: {str(e)}")
        return False

