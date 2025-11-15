# 🚀 Installation Guide

## System Requirements

### Minimum Requirements

- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB free space
- **Internet**: Required for initial setup and package installation

### Recommended Requirements

- **RAM**: 8GB or more
- **GPU**: NVIDIA GPU with CUDA support (for faster model inference)
- **Disk Space**: 5GB for datasets and models

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning.git
cd Disease-prediction-using-Machine-Learning
```

### 2. Set Up Python Environment

#### Windows

```powershell
# Create virtual environment
python -m venv env

# Activate virtual environment
.\env\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

#### macOS/Linux

```bash
# Create virtual environment
python3 -m venv env

# Activate virtual environment
source env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

If you encounter issues, install packages individually:

```bash
pip install streamlit==1.48.1
pip install tensorflow==2.15.0
pip install scikit-learn==1.3.2
pip install pandas numpy pillow
pip install plotly bcrypt pymongo sqlalchemy python-dotenv fpdf
```

### 4. Database Configuration

#### Option A: MongoDB (Recommended)

1. Install MongoDB:

   - **Windows**: Download from [mongodb.com](https://www.mongodb.com/try/download/community)
   - **macOS**: `brew install mongodb-community`
   - **Linux**: Follow [official guide](https://docs.mongodb.com/manual/installation/)

2. Start MongoDB service:

   ```bash
   # Windows (run as Administrator)
   net start MongoDB

   # macOS
   brew services start mongodb-community

   # Linux
   sudo systemctl start mongod
   ```

3. Create database:
   ```bash
   mongosh
   use medical_diagnosis
   ```

#### Option B: PostgreSQL

1. Install PostgreSQL
2. Create database:
   ```sql
   CREATE DATABASE medical_diagnosis;
   ```

#### Option C: SQLite (Quick Start)

No installation needed - just use in .env:

```env
DATABASE_URL=sqlite:///medical_diagnosis.db
```

### 5. Environment Configuration

1. Copy the example environment file:

   ```bash
   cp config/.env.example .env
   ```

2. Edit `.env` file with your configuration:

   ```env
   # MongoDB
   DATABASE_URL=mongodb://localhost:27017/medical_diagnosis

   # PostgreSQL
   # DATABASE_URL=postgresql://username:password@localhost:5432/medical_diagnosis

   # MySQL
   # DATABASE_URL=mysql://username:password@localhost:3306/medical_diagnosis

   # SQLite (for testing)
   # DATABASE_URL=sqlite:///medical_diagnosis.db
   ```

### 6. Initialize Database Tables

```bash
python scripts/setup_auth.py
```

This will create necessary tables for user authentication and prediction storage.

### 7. Download Model Files

**Option A**: Train your own models (recommended)

```bash
python scripts/train_diabetes.py
python scripts/train_kidney.py
python scripts/train_lung.py
python scripts/train_breast_cancer_image.py
```

**Option B**: Use pre-trained models (if available)

- Place model files in `saved_models/` directory
- Required files:
  - `lung_disease_model.h5`
  - `diabetes_model.joblib`
  - `diabetes_scaler.joblib`
  - `kidney_model.joblib`
  - `breast_cancer_model.h5`

### 8. Run the Application

```bash
streamlit run Home.py
```

The application will open automatically at `http://localhost:8501`

## Verification

Test the installation:

1. Open browser to `http://localhost:8501`
2. Register a new account
3. Login with your credentials
4. Try any disease module
5. Check if predictions work

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

#### 2. TensorFlow GPU Issues

```bash
# Install CPU version instead
pip uninstall tensorflow
pip install tensorflow-cpu
```

#### 3. Database Connection Error

- Check if database service is running
- Verify DATABASE_URL in .env file
- Check firewall settings

#### 4. Port Already in Use

```bash
# Run on different port
streamlit run Home.py --server.port 8502
```

#### 5. Memory Issues

- Close other applications
- Reduce model batch size
- Use CPU instead of GPU

### Getting Help

- Check [GitHub Issues](https://github.com/Kesamreddyprashanthreddy/Disease-prediction-using-Machine-Learning/issues)
- Read [Documentation](docs/)
- Contact: kesamreddyprashanthreddy@example.com

## Next Steps

After successful installation:

1. Read the [User Guide](docs/USER_GUIDE.md)
2. Explore the [API Documentation](docs/API.md)
3. Check [Contributing Guidelines](CONTRIBUTING.md)
4. Review [Security Best Practices](docs/SECURITY.md)

## Uninstallation

To remove the application:

```bash
# Deactivate virtual environment
deactivate

# Remove project directory
cd ..
rm -rf Disease-prediction-using-Machine-Learning

# Remove database (optional)
# MongoDB: drop medical_diagnosis database
# PostgreSQL: DROP DATABASE medical_diagnosis;
```

---

**✅ Installation Complete!** You're ready to use the AI Medical Diagnosis System.
