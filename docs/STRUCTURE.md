# 📂 Project Structure

## Overview

This document describes the organization of the Disease Prediction System codebase.

## Directory Structure

```
Disease-prediction-using-Machine-Learning/
│
├── 📄 Home.py                          # Main application entry point
├── 📄 requirements.txt                 # Python dependencies
├── 📄 README.md                        # Project documentation
├── 📄 LICENSE                          # MIT License
├── 📄 CONTRIBUTING.md                  # Contribution guidelines
├── 📄 .gitignore                       # Git ignore rules
├── 📄 .env                             # Environment variables (not in Git)
│
├── 📁 src/                             # Source code directory
│   ├── 📄 __init__.py                  # Package initializer
│   ├── 📄 auth.py                      # Authentication module
│   ├── 📄 database.py                  # Database operations
│   ├── 📄 utils.py                     # Utility functions
│   │
│   └── 📁 pages/                       # Streamlit multi-page app
│       ├── 📄 1_🫁_Lung_Disease.py     # Lung disease module
│       ├── 📄 2_🩺_Diabetes.py         # Diabetes risk module
│       ├── 📄 3_🫘_Kidney_Disease.py   # Kidney disease module
│       ├── 📄 4_🎗️_Breast_Cancer.py   # Breast cancer module
│       ├── 📄 5_📊_Prediction_History.py # History viewer
│       ├── 📄 _Login.py                # Login page
│       └── 📄 _Register.py             # Registration page
│
├── 📁 pages/                           # Symlink/copy for Streamlit
│   └── (same as src/pages/)
│
├── 📁 data/                            # Data directory
│   ├── 📁 raw/                         # Raw datasets
│   │   ├── diabetes_disease.csv
│   │   └── kidney_disease.csv
│   └── 📁 processed/                   # Processed data
│
├── 📁 saved_models/                    # Trained models
│   ├── lung_disease_model.h5
│   ├── diabetes_model.joblib
│   ├── diabetes_scaler.joblib
│   ├── kidney_model.joblib
│   └── breast_cancer_model.h5
│
├── 📁 scripts/                         # Utility scripts
│   ├── 📄 train_lung.py                # Train lung model
│   ├── 📄 train_diabetes.py            # Train diabetes model
│   ├── 📄 train_kidney.py              # Train kidney model
│   ├── 📄 train_breast_cancer_image.py # Train breast cancer model
│   ├── 📄 setup_auth.py                # Initialize authentication
│   ├── 📄 setup_mongodb.py             # MongoDB setup
│   ├── 📄 run_app.bat                  # Windows run script
│   ├── 📄 run_app.sh                   # Linux/Mac run script
│   ├── 📄 start.bat                    # Windows start script
│   └── 📄 start.sh                     # Linux/Mac start script
│
├── 📁 tests/                           # Unit tests
│   ├── 📄 test.py                      # Basic tests
│   ├── 📄 unit_tests.py                # Comprehensive tests
│   ├── 📄 test_cases_results.py        # Test results
│   └── 📄 final_testing_validation.py  # Final validation
│
├── 📁 config/                          # Configuration files
│   └── 📄 .env.example                 # Environment template
│
├── 📁 docs/                            # Documentation
│   ├── 📄 INSTALL.md                   # Installation guide
│   ├── 📄 API.md                       # API documentation
│   ├── 📄 USER_GUIDE.md                # User manual
│   └── 📄 SECURITY.md                  # Security guidelines
│
├── 📁 .streamlit/                      # Streamlit configuration
│   └── 📄 config.toml                  # Streamlit settings
│
├── 📁 archive/                         # Old/deprecated files
│   ├── app.py
│   ├── main_new.py
│   └── (other old files)
│
├── 📁 env/                             # Virtual environment (not in Git)
│
└── 📁 __pycache__/                     # Python cache (not in Git)
```

## Module Descriptions

### Core Application Files

#### `Home.py`

- Main entry point for the Streamlit application
- Handles navigation between disease modules
- Displays welcome screen and module selection
- Manages user authentication UI

#### `src/auth.py`

- User authentication and session management
- Password hashing with bcrypt
- Login/logout functionality
- Session persistence

#### `src/database.py`

- Database connection management
- Multi-database support (MongoDB, PostgreSQL, MySQL, SQLite)
- User operations (CRUD)
- Prediction result storage and retrieval

#### `src/utils.py`

- Common utility functions
- Data preprocessing helpers
- Model loading utilities
- Shared helper functions

### Disease Modules

Each disease module follows a similar structure:

1. **Model Loading**: Load pre-trained ML models
2. **Data Input**: File upload or manual entry
3. **Preprocessing**: Data cleaning and transformation
4. **Prediction**: ML model inference
5. **Visualization**: Interactive charts and graphs
6. **Results**: Detailed medical interpretation
7. **Reporting**: PDF report generation
8. **Database**: Save results to history

### Data Organization

#### `data/raw/`

- Original, unmodified datasets
- CSV files for diabetes and kidney disease
- Medical images (chest X-rays, mammograms)

#### `data/processed/`

- Cleaned and preprocessed data
- Feature-engineered datasets
- Split datasets (train/test/validation)

### Scripts

#### Training Scripts

- `train_lung.py`: Train CNN for lung disease detection
- `train_diabetes.py`: Train KNN for diabetes prediction
- `train_kidney.py`: Train Random Forest for kidney disease
- `train_breast_cancer_image.py`: Train VGG16 for breast cancer

#### Setup Scripts

- `setup_auth.py`: Initialize database tables
- `setup_mongodb.py`: Configure MongoDB collections
- `run_app.bat`/`run_app.sh`: Application launchers

### Testing

#### Test Organization

- `test.py`: Basic functionality tests
- `unit_tests.py`: Comprehensive unit tests
- `test_cases_results.py`: Test result logging
- `final_testing_validation.py`: Pre-deployment validation

## Configuration Files

### `.env`

Environment variables for sensitive data:

```env
DATABASE_URL=mongodb://localhost:27017/medical_diagnosis
SECRET_KEY=your-secret-key
DEBUG=False
```

### `.streamlit/config.toml`

Streamlit application settings:

```toml
[server]
port = 8501
enableCORS = false

[theme]
primaryColor = "#667eea"
backgroundColor = "#f5f7fa"
```

## Import Structure

### Python Path Setup

All modules add `src/` to Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
```

### Import Examples

```python
# From disease modules
from auth import auth
from database import get_prediction_operations
from utils import preprocess_image

# Within src
from src.auth import auth
from src.database import get_user_operations
```

## File Naming Conventions

### Python Files

- `snake_case.py` for regular modules
- `1_🫁_Module.py` for Streamlit pages (numbered + emoji)
- `_Login.py` for hidden pages (underscore prefix)

### Data Files

- `disease_name_data.csv` for datasets
- `model_name_model.h5` for Keras models
- `model_name_model.joblib` for scikit-learn models

### Documentation

- `UPPERCASE.md` for root-level docs (README, LICENSE)
- `PascalCase.md` for detailed docs in `docs/`

## Dependency Management

### Core Dependencies

```
streamlit        # Web framework
tensorflow       # Deep learning
scikit-learn     # Classical ML
pandas/numpy     # Data manipulation
```

### Database Drivers

```
pymongo         # MongoDB
sqlalchemy      # SQL databases
```

### Security

```
bcrypt          # Password hashing
python-dotenv   # Environment variables
```

## Best Practices

### Code Organization

1. Keep modules focused and single-purpose
2. Use clear, descriptive names
3. Add docstrings to all functions
4. Group related functionality

### File Management

1. Keep large data files out of Git
2. Use `.gitignore` properly
3. Archive deprecated code
4. Document file purposes

### Version Control

1. Commit frequently with clear messages
2. Use branches for features
3. Keep commits atomic
4. Review before merging

## Getting Started

1. **Installation**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**

   ```bash
   cp config/.env.example .env
   # Edit .env with your settings
   ```

3. **Database Setup**

   ```bash
   python scripts/setup_auth.py
   ```

4. **Run Application**
   ```bash
   streamlit run Home.py
   ```

## Maintenance

### Regular Tasks

- Update dependencies monthly
- Review and clean logs
- Backup database weekly
- Monitor disk space
- Check security updates

### Code Quality

- Run tests before commits
- Use linting tools
- Review code style
- Update documentation

---

**Last Updated**: November 2025
**Maintainer**: Kesamreddy Prashant Reddy
