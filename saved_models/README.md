# Trained Models Directory

## 📦 Model Files

The trained model files are **not included in this repository** due to their large size (500+ MB total).

## 🔧 How to Get Models

### Option 1: Train Models Yourself (Recommended)

Train all models using the provided scripts:

```bash
# Train Lung Disease Model (CNN)
python scripts/train_lung.py

# Train Diabetes Model (KNN)
python scripts/train_diabetes.py

# Train Kidney Disease Model (Random Forest)
python scripts/train_kidney.py

# Train Breast Cancer Model (VGG16)
python scripts/train_breast_cancer_image.py
```

**Training Time:** 30 minutes - 2 hours depending on your hardware

### Option 2: Download Pre-trained Models

Contact the repository owner for access to pre-trained models:

- Email: [Add your email]
- Or open an issue requesting model files

### Option 3: Use Git LFS (Advanced)

If you have Git LFS installed:

```bash
git lfs pull
```

## 📋 Expected Model Files

After training, you should have these files:

- `lung_disease_model.h5` (~287 MB) - CNN for chest X-rays
- `breast_cancer_image_model_improved.h5` (~164 MB) - VGG16 for mammograms
- `breast_cancer_image_model.h5` (~84 MB) - Alternative breast cancer model
- `best_breast_model.h5` (~60 MB) - Best performing breast cancer model
- `diabetes_model.joblib` - KNN classifier for diabetes
- `diabetes_scaler.joblib` - Feature scaler for diabetes
- `kidney_model.joblib` - Random Forest for kidney disease
- `kidney_scaler.joblib` - Feature scaler for kidney disease

## ⚠️ Important Notes

- **Do not commit large model files to GitHub** (they exceed 100 MB limit)
- Models are platform-independent (work on Windows/Mac/Linux)
- Trained models maintain ~90-95% accuracy on test data
- If training, ensure you have the required datasets in `data/raw/`

## 🆘 Troubleshooting

**Model not found error?**

- Make sure you've trained the models or downloaded them
- Check that files are in the `saved_models/` directory
- Verify file permissions (models should be readable)

**Out of memory during training?**

- Reduce batch size in training scripts
- Close other applications
- Consider using Google Colab for training

## 📚 Additional Resources

- [Installation Guide](../docs/INSTALL.md)
- [Project Structure](../docs/STRUCTURE.md)
- [Contributing Guidelines](../CONTRIBUTING.md)
