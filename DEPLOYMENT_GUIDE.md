# 🚀 Hugging Face Spaces Deployment Guide

## Prerequisites

✅ GitHub account with your code pushed
✅ Hugging Face account (free)
✅ All model files in `saved_models/` directory

## Step-by-Step Deployment

### 1️⃣ Create a New Space

1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in details:
   - **Space name**: `ai-medical-diagnosis`
   - **License**: MIT
   - **SDK**: Streamlit
   - **Hardware**: CPU (free) or upgrade to GPU if needed
   - **Visibility**: Public or Private

### 2️⃣ Connect to GitHub

**Option A: Import from GitHub (Recommended)**

1. In Space settings, go to **"Files and versions"**
2. Click **"Import from GitHub"**
3. Enter your repo: `Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning`
4. Select branch: `main`
5. Click **"Import"**

**Option B: Manual Upload**

1. Clone your repo locally
2. In HF Space, click **"Files"** → **"Add file"** → **"Upload files"**
3. Upload all project files maintaining folder structure

### 3️⃣ Required Files Checklist

Ensure these files are in your Space:

```
✅ Home.py (main entry point)
✅ requirements.txt (dependencies)
✅ README.md (with HF metadata at top)
✅ .streamlit/config.toml (Streamlit configuration)
✅ pages/ (all disease modules)
✅ src/ (auth.py, database.py, utils.py)
✅ saved_models/ (all .h5 and .joblib files)
```

### 4️⃣ Configure Environment Variables (Optional)

If using MongoDB or other services:

1. Go to Space **Settings** → **"Variables and secrets"**
2. Add secrets:
   - `MONGODB_URI` = your MongoDB connection string
   - `DATABASE_URL` = your database URL (if any)

### 5️⃣ Verify README.md Header

Your README.md should have this at the top:

```yaml
---
title: AI Medical Diagnosis System
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.48.1
app_file: Home.py
pinned: false
license: mit
python_version: "3.10"
---
```

### 6️⃣ Build & Deploy

1. HF Spaces will automatically detect Streamlit
2. It will install dependencies from `requirements.txt`
3. Build takes 5-10 minutes
4. Watch the **"Logs"** tab for any errors

### 7️⃣ Test Your App

Once deployed:

1. Your app URL: `https://huggingface.co/spaces/YOUR_USERNAME/ai-medical-diagnosis`
2. Test all features:
   - ✅ User registration/login
   - ✅ Each disease module
   - ✅ File uploads
   - ✅ PDF report generation
   - ✅ Prediction history

---

## 🐛 Common Issues & Solutions

### Issue 1: Model Files Too Large

**Error**: "File size exceeds limit"
**Solution**:

- Use Git LFS for large files:

```bash
git lfs install
git lfs track "*.h5"
git lfs track "*.joblib"
git add .gitattributes
git add saved_models/
git commit -m "Add models with Git LFS"
git push
```

### Issue 2: Import Errors

**Error**: "ModuleNotFoundError"
**Solution**:

- Check `requirements.txt` has all dependencies
- Verify Python version compatibility
- Add missing packages

### Issue 3: Database Connection Failed

**Error**: "Cannot connect to MongoDB"
**Solution**:

- Use MongoDB Atlas (free tier)
- Add connection string as Space secret
- Update `database.py` to use environment variable

### Issue 4: Port Already in Use

**Error**: "Port 8501 is already in use"
**Solution**:

- Hugging Face uses port 7860
- Ensure `.streamlit/config.toml` has `port = 7860`

### Issue 5: Memory Error

**Error**: "Out of memory"
**Solution**:

- Upgrade to better hardware (CPU upgrade or GPU)
- Optimize model loading with @st.cache_resource
- Reduce batch sizes

---

## 🎯 Post-Deployment

### Enable Better Hardware (if needed)

1. Go to Space **Settings** → **"Hardware"**
2. Upgrade from CPU basic to:
   - CPU upgrade ($0.03/hour) - Better performance
   - Tesla T4 Small ($0.60/hour) - GPU support
   - Tesla T4 Medium ($1.20/hour) - More GPU memory

### Share Your Space

Your app URL: `https://huggingface.co/spaces/YOUR_USERNAME/ai-medical-diagnosis`

### Embed in Website

```html
<iframe
  src="https://YOUR_USERNAME-ai-medical-diagnosis.hf.space"
  frameborder="0"
  width="100%"
  height="800"
></iframe>
```

---

## 📊 Monitoring

### Check Logs

- Click **"Logs"** tab to see real-time application logs
- Monitor for errors or warnings
- Check performance metrics

### Update Your Space

To update after pushing to GitHub:

1. Go to Space **Settings**
2. Click **"Sync with GitHub"**
3. Or manually upload changed files

---

## 🔒 Security Notes

⚠️ **Important**: Don't commit sensitive data!

- Use Spaces secrets for API keys
- Don't hardcode passwords in code
- Use environment variables for credentials

---

## 📚 Additional Resources

- [HF Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Streamlit on HF Guide](https://huggingface.co/docs/hub/spaces-sdks-streamlit)
- [Git LFS Setup](https://git-lfs.github.com/)

---

## 🎉 Success!

Your AI Medical Diagnosis System is now live on Hugging Face Spaces!

**Next Steps:**

1. Test all features thoroughly
2. Share the link with users
3. Monitor logs for issues
4. Update models as needed
5. Consider adding to Hugging Face Hub for visibility

---

**Need Help?**

- HF Discord: https://discord.gg/hugging-face
- HF Forums: https://discuss.huggingface.co/
