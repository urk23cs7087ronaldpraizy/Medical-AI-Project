<<<<<<< HEAD
# MedVision AI — Medical Image Diagnosis Support System

## 🏥 AI-Based Medical Image Classification and Diagnosis Support System

A complete AI-powered healthcare application that analyzes medical images (X-ray, MRI, CT scans) to detect diseases, classify abnormalities, explain diagnoses using Grad-CAM, and recommend nearby hospitals.

---

## 📋 Project Overview

| Feature | Description |
|---------|-------------|
| **Multi-Disease Detection** | Pneumonia, Brain Tumor, Bone Fracture, Normal |
| **AI Model** | MobileNetV2 (Transfer Learning) with TensorFlow/Keras |
| **Explainable AI** | Grad-CAM heatmap visualization |
| **Medical Knowledge** | Disease causes, symptoms, risk factors, prevention |
| **Hospital Finder** | Location-based hospital recommendations |
| **Web Dashboard** | Professional Flask + Bootstrap UI |

---

## 🗂️ Project Structure

```
medical-ai-project/
├── run.py                          # Main entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── dataset/
│   ├── __init__.py
│   ├── download_dataset.py         # Dataset download & synthetic generation
│   └── processed/                  # Generated dataset (train/test splits)
│
├── models/
│   ├── __init__.py
│   ├── train_model.py              # Model training with MobileNetV2
│   ├── model.h5                    # Saved trained model
│   ├── class_mapping.json          # Class label mapping
│   └── training_results/           # Training history plots
│
├── backend/
│   ├── __init__.py
│   ├── app.py                      # Flask REST API server
│   ├── predict.py                  # Prediction engine + Grad-CAM
│   └── utils.py                    # Medical knowledge base + hospital DB
│
├── frontend/
│   ├── templates/
│   │   └── index.html              # Main dashboard HTML
│   └── static/
│       ├── css/
│       │   └── style.css           # Professional medical UI styles
│       └── js/
│           └── app.js              # Frontend logic & API integration
│
├── uploads/                        # Uploaded medical images
└── gradcam_outputs/                # Generated Grad-CAM visualizations
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
cd medical-ai-project
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python run.py
```

The system will:
1. Auto-build a demo model (MobileNetV2) if none exists
2. Start the Flask web server on `http://localhost:5000`

### Step 3: Open the Dashboard

Open your browser and navigate to: **http://localhost:5000**

---

## 📖 Detailed Setup

### Generate Training Dataset

```bash
python run.py --dataset
```

This generates synthetic medical images for 4 categories (250 images each).

### Train the AI Model

```bash
python run.py --train
```

Trains MobileNetV2 with transfer learning on the medical dataset.

### Full Setup (Dataset + Model)

```bash
python run.py --setup
```

---

## 🧠 AI Model Architecture

```
Input (224x224x3) → MobileNetV2 (ImageNet pretrained, frozen)
                   → Global Average Pooling
                   → BatchNormalization
                   → Dense(256, ReLU) → Dropout(0.5)
                   → BatchNormalization
                   → Dense(128, ReLU) → Dropout(0.3)
                   → Dense(4, Softmax) → Output
```

**Training Strategy:**
1. Phase 1: Train classification head (base frozen)
2. Phase 2: Fine-tune last 30 layers of MobileNetV2

---

## 🔥 Grad-CAM Explainability

The system implements Gradient-weighted Class Activation Mapping (Grad-CAM) to provide visual explanations:

- Highlights which regions of the image influenced the prediction
- Generates overlay heatmaps on the original image
- Provides human-readable explanations for each diagnosis

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Main dashboard |
| `POST` | `/api/predict` | Upload image & get prediction |
| `GET` | `/api/gradcam/<filename>` | Get Grad-CAM visualization |
| `GET` | `/api/history` | Get prediction history |
| `GET` | `/api/disease-info/<type>` | Get disease information |
| `GET` | `/api/hospitals` | Get hospital recommendations |
| `GET` | `/api/system-info` | Get system status |

---

## 🏥 Supported Disease Categories

| Category | Image Type | Description |
|----------|-----------|-------------|
| **Normal** | X-ray/MRI | No abnormalities detected |
| **Pneumonia** | Chest X-ray | Lung infection/inflammation |
| **Tumor** | Brain MRI | Abnormal cell growth |
| **Fracture** | Bone X-ray | Bone discontinuity |

---

## 🔧 Technologies Used

| Component | Technology |
|-----------|-----------|
| **AI Framework** | TensorFlow 2.x / Keras |
| **Model Architecture** | MobileNetV2 (Transfer Learning) |
| **Explainability** | Grad-CAM |
| **Backend** | Python Flask |
| **Frontend** | HTML5, CSS3, JavaScript, Bootstrap 5 |
| **Image Processing** | OpenCV, Pillow |
| **Data Science** | NumPy, Matplotlib, scikit-learn |

---

## ⚠️ Disclaimer

This is a **decision-support tool** for healthcare professionals developed as an academic project. It is **NOT** intended to replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

---

## 📄 License

This project is developed for educational and research purposes as a Final Year Major Project.
=======
# Medical-AI-Project
>>>>>>> origin/main
