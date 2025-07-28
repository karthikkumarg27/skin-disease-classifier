Skin Disease Classification Project
A machine learning project for classifying skin lesions using the HAM10000 dataset. This project implements deep learning models to assist in the diagnosis of various skin conditions including melanoma, basal cell carcinoma, and other dermatological conditions.
📁 Project Structure
SKIN_DISEASE/
├── app/
│   └── streamlit_app.py          # Streamlit web application
├── data/
│   ├── HAM10000_images/          # Image dataset folder
│   └── HAM10000_metadata.csv     # Dataset metadata
├── src/
│   ├── __pycache__/              # Python cache files
│   ├── dataloader.py             # Data loading and preprocessing
│   ├── evaluate.py               # Model evaluation utilities
│   ├── model.py                  # Neural network model definitions
│   └── train.py                  # Training pipeline
├── venv/                         # Virtual environment
├── .gitignore                    # Git ignore file
├── .python-version               # Python version specification
├── best_model.h5                 # Best trained model weights
├── final_model.h5                # Final model weights
└── requirements.txt              # Project dependencies
🎯 Project Overview
This project aims to classify skin lesions into different categories using computer vision and deep learning techniques. The classification can help dermatologists and healthcare professionals in early detection and diagnosis of skin conditions.
Key Features

Deep learning-based skin lesion classification
Interactive web application using Streamlit
Comprehensive data preprocessing pipeline
Model evaluation and performance metrics
Pre-trained model weights for quick deployment

📊 Dataset
HAM10000 Dataset
The project uses the HAM10000 (Human Against Machine with 10000 training images) dataset, which contains dermatoscopic images of common pigmented skin lesions.
Dataset Categories:

Melanoma (mel)
Melanocytic nevus (nv)
Basal cell carcinoma (bcc)
Actinic keratosis / Bowen's disease (akiec)
Benign keratosis (bkl)
Dermatofibroma (df)
Vascular lesion (vasc)

📥 Dataset Download
To download the dataset:

Option 1: Direct Download

Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download
Click "Download" button (requires Kaggle account)


Option 2: Using Kaggle API
bash# Install Kaggle API
pip install kaggle

# Configure API credentials (place kaggle.json in ~/.kaggle/)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000

# Extract the dataset
unzip skin-cancer-mnist-ham10000.zip -d data/

Option 3: Using Python Script
pythonimport kaggle

# Download dataset
kaggle.api.dataset_download_files(
    'kmader/skin-cancer-mnist-ham10000', 
    path='data/', 
    unzip=True
)


After downloading:

Place images in data/HAM10000_images/
Place metadata CSV in data/HAM10000_metadata.csv