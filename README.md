# Brain_tumor_detection_and_classification
# 🧠 Brain Tumor MRI Classification

> A deep learning-based system for automated brain tumor classification from MRI scans using transfer learning and ensemble learning.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Neural%20Networks-red?logo=keras)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## 📖 About

This project focuses on the automatic classification of brain tumors from MRI images using deep learning. Three pretrained CNN architectures—**DenseNet201**, **InceptionV3**, and **EfficientNetB3**—are fine-tuned through transfer learning and combined using an **Ensemble Voting Model** to improve prediction accuracy and reliability.

The system classifies MRI images into four categories:

- 🟢 No Tumor
- 🔴 Glioma
- 🟡 Meningioma
- 🔵 Pituitary Tumor

---

## 🎯 Objectives

- Develop a reliable AI model for brain tumor classification.
- Compare multiple pretrained CNN architectures.
- Improve performance through ensemble learning.
- Evaluate models using standard classification metrics.

---

## 🏗️ Model Architecture

```
MRI Images
     │
     ▼
Image Preprocessing
     │
     ▼
Transfer Learning
 ├── DenseNet201
 ├── InceptionV3
 └── EfficientNetB3
     │
     ▼
Ensemble Voting
     │
     ▼
Final Tumor Prediction
```

---

## ⚙️ Technologies

| Category | Tools |
|----------|-------|
| Language | Python |
| Framework | TensorFlow, Keras |
| Image Processing | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib |
| Evaluation | Scikit-learn |

---

## 📊 Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

---

## 📁 Repository Structure

```
📦 Brain-Tumor-Classification
├── 📂 Dataset
├── 📂 Models
├── 📂 Training
├── 📂 Evaluation
├── 📂 Results
├── 📂 Saved Models
├── 📄 train.py
├── 📄 predict.py
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🚀 Workflow

1. Collect and preprocess MRI images.
2. Train DenseNet201, InceptionV3, and EfficientNetB3.
3. Evaluate each model individually.
4. Combine predictions using an Ensemble Voting Model.
5. Compare results and generate performance metrics.

---

## 📈 Highlights

✅ Transfer Learning

✅ Multiclass Classification

✅ Ensemble Learning

✅ High Performance CNN Models

✅ Comprehensive Model Evaluation

---

## 💡 Future Scope

- Explainable AI using Grad-CAM
- Web Application Deployment
- Real-time MRI Prediction
- Hyperparameter Optimization
- Cloud Deployment

---



### ⭐ If you like this project, don't forget to star the repository!
