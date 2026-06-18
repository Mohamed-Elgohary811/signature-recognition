# signature-recognition-system
# AI Signature Verification System

A deep learning-based signature verification system that classifies handwritten signatures as **Genuine** or **Forged** using Transfer Learning with **ResNet-34** and a modular Machine Learning pipeline.

The project is designed with a production-oriented structure that separates data ingestion, preprocessing, model training, evaluation, model management, and inference into independent pipeline stages.

---

## Project Overview

Signature verification plays an important role in fraud detection, banking systems, document authentication, insurance claims, and identity verification.

This project implements an end-to-end workflow that allows users to:

* Train a deep learning model on handwritten signature images.
* Evaluate model performance on unseen data.
* Manage model artifacts and versions.
* Serve predictions through a FastAPI backend.
* Upload signature images through a web interface and receive real-time predictions.

---

## Features

* Modular ML Pipeline Architecture
* Configuration-Driven Workflow (YAML)
* Automated Data Ingestion
* Image Preprocessing & Augmentation
* Transfer Learning with ResNet-34
* Model Evaluation & Best Model Selection
* Artifact Management
* Custom Logging & Exception Handling
* FastAPI REST API
* Interactive Web Interface
* Real-Time Signature Classification
* Confidence Score Visualization

---

## Dataset

This project uses the Handwritten Signature Verification Dataset available on Kaggle:

Dataset:
https://www.kaggle.com/datasets/tienen/handwritten-signature-verification

Dataset Statistics:

* 5,626 signature samples
* Genuine signatures
* Forged signatures
* 275 unique original signers
* 247 forged signer identities

Classes:

```text
real
forged
```

---

## Project Architecture

```text
Signature Recognition System
│
├── Data Ingestion
│
├── Data Transformation
│
├── Model Training
│
├── Model Evaluation
│
├── Model Pusher
│
├── Prediction Pipeline
│
└── FastAPI Application
```

---

## Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Ingestion
   │
   ▼
Data Transformation
   │
   ▼
Train / Validation / Test Split
   │
   ▼
ResNet34 Transfer Learning
   │
   ▼
Model Evaluation
   │
   ▼
Best Model Selection
   │
   ▼
Prediction API
   │
   ▼
Web Application
```

---

## Image Preprocessing

The following preprocessing steps are applied:

* Resize Images to 224 × 224
* Random Rotation Augmentation
* Tensor Conversion
* ImageNet Normalization

Normalization:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## Model

Base Model:

```text
ResNet-34 (Pretrained)
```

Training Strategy:

* Transfer Learning
* Cross Entropy Loss
* SGD Optimizer with Momentum
* Binary Classification

Output Classes:

```text
Genuine Signature
Forged Signature
```

---

## Tech Stack

### Machine Learning

* PyTorch
* TorchVision
* NumPy
* Pandas

### Backend

* FastAPI
* Uvicorn

### Image Processing

* OpenCV
* Pillow

### Configuration

* PyYAML

### Frontend

* HTML
* CSS
* JavaScript

---

## API Endpoints

### Start Training

```http
POST /train
```

### Check Training Status

```http
GET /training-status
```

### Predict Signature

```http
POST /predict
```

### API Documentation

```http
GET /docs
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Mohamed-Elgohary811/signature-recognition.git

cd signature-recognition
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
python app.py
```

or

```bash
uvicorn app:app --reload
```

Application:

```text
http://127.0.0.1:8001
```

API Documentation:

```text
http://127.0.0.1:8001/docs
```

---

## Future Improvements

* Model Versioning
* Docker Containerization
* CI/CD Integration
* Cloud Deployment
* Experiment Tracking
* Advanced Signature Verification Models
* Explainable AI (XAI)
* User Authentication System

---

## Repository

GitHub Repository:

https://github.com/Mohamed-Elgohary811/signature-recognition

---

## Author

Mohamed Elgohary

Machine Learning Engineer | AI Enthusiast

Feel free to connect, provide feedback, or contribute to the project.
[GitHub](https://github.com/Mohamed-Elgohary811) | [LinkedIn](https://www.linkedin.com/feed/)
