# 🚀 Streamlit Cloud Deployment Guide

## ⚠️ Important: Streamlit Cloud Does NOT Support Git LFS

Your large model files (`.h5`, `.joblib`) won't be downloaded automatically. You need to use cloud storage.

---

## ✅ Solution Options (Choose ONE)

### **Option 1: GitHub Releases** (Recommended - Free & Easy)

#### Step 1: Create a Release with Model Files

1. Go to your GitHub repo: https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning
2. Click **"Releases"** (right sidebar)
3. Click **"Create a new release"**
4. Tag: `v1.0` or `models-v1`
5. Title: `Model Files`
6. Click **"Attach binaries"** and upload:
   - `lung_disease_model.h5`
   - `breast_cancer_image_model_improved.h5`
   - `diabetes_model_optimized.joblib`
   - `diabetes_scaler.joblib`
   - `kidney_model.joblib`

7. Click **"Publish release"**

#### Step 2: Get Download URLs

After publishing, right-click each file → **"Copy link address"**

Example URL format:
```
https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning/releases/download/v1.0/lung_disease_model.h5
```

#### Step 3: Update `download_models.py`

Edit line 54 in `download_models.py`:

```python
models_config = {
    "saved_models/lung_disease_model.h5": 
        "https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning/releases/download/v1.0/lung_disease_model.h5",
    
    "saved_models/breast_cancer_image_model_improved.h5":
        "https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning/releases/download/v1.0/breast_cancer_image_model_improved.h5",
    
    "saved_models/diabetes_model_optimized.joblib":
        "https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning/releases/download/v1.0/diabetes_model_optimized.joblib",
    
    "saved_models/diabetes_scaler.joblib":
        "https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning/releases/download/v1.0/diabetes_scaler.joblib",
    
    "saved_models/kidney_model.joblib":
        "https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning/releases/download/v1.0/kidney_model.joblib",
}
```

---

### **Option 2: Hugging Face Hub** (Best for ML Models)

#### Step 1: Create Model Repository

1. Go to https://huggingface.co/new
2. Create repository: `disease-prediction-models`
3. Upload your model files

#### Step 2: Get URLs

Format: `https://huggingface.co/YOUR_USERNAME/disease-prediction-models/resolve/main/MODEL_FILE.h5`

#### Step 3: Update download_models.py

```python
models_config = {
    "saved_models/lung_disease_model.h5": 
        "https://huggingface.co/YOUR_USERNAME/disease-prediction-models/resolve/main/lung_disease_model.h5",
    # ... add other models
}
```

---

### **Option 3: Google Drive** (Simple but Slower)

#### Step 1: Upload to Google Drive

1. Upload model files to Google Drive
2. Right-click → Share → **"Anyone with the link"**
3. Copy the sharing link

#### Step 2: Get File ID

From link: `https://drive.google.com/file/d/FILE_ID_HERE/view?usp=sharing`

Extract the `FILE_ID_HERE` part

#### Step 3: Install gdown

Add to `requirements.txt`:
```
gdown>=4.7.1
```

#### Step 4: Use in Code

```python
from download_models import download_from_google_drive

download_from_google_drive(
    file_id="YOUR_FILE_ID",
    destination="saved_models/lung_disease_model.h5",
    file_name="Lung Disease Model"
)
```

---

## 🔄 Update Model Loading Functions

Add this at the start of each model loading function:

```python
# In pages/1_🫁_Lung_Disease.py
from download_models import ensure_models_downloaded

@st.cache_resource
def load_lung_model():
    # Download models if missing
    ensure_models_downloaded()
    
    # Rest of your loading code...
```

---

## 📦 Deploy to Streamlit Cloud

1. Push all changes to GitHub
2. Go to https://share.streamlit.io/
3. Click **"New app"**
4. Select your repo
5. Main file: `Home.py`
6. Click **"Deploy"**

### Add Secrets

In Streamlit Cloud dashboard:
- Go to **Settings** → **Secrets**
- Add:
```toml
DATABASE_URL = "sqlite:///disease_prediction.db"
```

---

## ✅ Verification

After deployment:
1. Check sidebar for "🔍 Model Debug"
2. Should show: `Size: 286.0 MB` (not 0.0 MB)
3. Models should load successfully

---

## 🆘 Troubleshooting

### Models still not loading?

1. **Check file sizes** in debug output
2. **Verify URLs** are publicly accessible (try in browser)
3. **Check requirements.txt** has `requests` package
4. **Look at Streamlit logs** for download errors

### Download timeout?

Increase timeout in `download_models.py`:
```python
response = requests.get(url, stream=True, timeout=600)  # 10 minutes
```

---

## 🎯 Quick Start (Recommended)

**Use GitHub Releases:**

1. Create release on GitHub
2. Upload all 5 model files
3. Copy download URLs
4. Update `download_models.py` with URLs
5. Push to GitHub
6. Deploy on Streamlit Cloud
7. ✅ Done!

Total time: ~15 minutes

---

Need help? The models will download automatically on first run when URLs are configured!

