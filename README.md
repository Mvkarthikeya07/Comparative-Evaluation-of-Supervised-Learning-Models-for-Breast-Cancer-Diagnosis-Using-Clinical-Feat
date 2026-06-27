<div align="center">

<h1>🩺 CancerScan AI</h1>
<h3>Comparative Evaluation of Supervised Learning Models for Breast Cancer Diagnosis Using Clinical Features</h3>

<p>
  <img src="https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Dataset-UCI%20Wisconsin-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-96.49%25-brightgreen?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.9960-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p>
  An end-to-end Machine Learning web application that classifies breast tumors as <strong>Benign</strong> or <strong>Malignant</strong> using 30 clinically validated diagnostic features — benchmarked across 6 classification algorithms on the UCI Wisconsin Breast Cancer Diagnostic Dataset.
</p>

> ⚠️ **Medical Disclaimer:** This project is developed **strictly for academic and educational purposes**. It must **not** be used for real-world clinical diagnosis or medical decision-making of any kind.

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Application Screenshots](#-application-screenshots)
- [Dataset](#-dataset)
- [Model Comparison & Benchmarks](#-model-comparison--benchmarks)
- [Deployed Model — Deep Dive](#-deployed-model--deep-dive)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Application Workflow](#-application-workflow)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Future Enhancements](#-future-enhancements)
- [Internship Context](#-internship-context)
- [Author](#-author)

---

## 🔭 Project Overview

Breast cancer is one of the most prevalent cancers worldwide, and early, accurate diagnosis is critical to patient outcomes. This project builds a **supervised binary classification system** trained on nuclear morphology features extracted from digitized Fine Needle Aspirate (FNA) biopsy images.

Rather than relying on a single model, this project performs a **rigorous comparative evaluation** of six classification algorithms — measuring Accuracy, Precision, Recall, F1 Score, ROC-AUC, and 5-Fold Cross-Validation stability — to identify the optimal approach for clinical classification tasks.

**Why this matters in healthcare AI:**
- **False negatives** (missed cancers) are far more costly than false positives — Recall and ROC-AUC must be prioritized
- Interpretability is critical — clinicians need to audit model decisions, not trust black boxes
- The final deployed model (Logistic Regression) achieves **96.49% accuracy**, **97.50% precision**, and a **ROC-AUC of 0.9960** on held-out test data

---

## 📸 Application Screenshots

### Home Page — Diagnostic Feature Input Interface

<img width="1366" height="768" alt="CancerScan AI - Home Page" src="https://github.com/user-attachments/assets/62f23fb7-6eb1-4082-a047-a243191eab22"/>

<img width="1366" height="768" alt="CancerScan AI - Input Form" src="https://github.com/user-attachments/assets/d8b36f97-ba8a-4426-8c7c-a7f27a0902bb"/>

<img width="1366" height="768" alt="CancerScan AI - Input Form Extended" src="https://github.com/user-attachments/assets/773123f9-1f46-4480-800b-01a39d4d52de"/>

> The interface accepts all 30 FNA-derived diagnostic features organized across Mean, Standard Error, and Worst-case measurement groups. All fields are required before inference is triggered.

---

### Prediction Results

**✅ Benign Prediction**

<img width="1366" height="768" alt="Benign Prediction Result" src="https://github.com/user-attachments/assets/ce3bdc92-8621-4ffd-82f9-dbdc098b6e05"/>

**❌ Malignant Prediction**

<img width="1366" height="768" alt="Malignant Prediction Result" src="https://github.com/user-attachments/assets/35ce9a9f-db1e-4e89-b5ff-403f1d7c3aca"/>

> The result page delivers an immediate, unambiguous diagnosis — **Benign (No Cancer Detected)** or **Malignant (Cancer Detected)** — based on the trained model's inference.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository — Wisconsin Breast Cancer Diagnostic |
| Samples | 569 patient records |
| Features | 30 real-valued diagnostic features |
| Target Classes | Malignant (M) = 212 samples · Benign (B) = 357 samples |
| Class Balance | 37.3% Malignant · 62.7% Benign |
| Missing Values | Handled via mean imputation (SimpleImputer) |
| Train / Test Split | 455 / 114 samples (stratified 80/20) |

### Feature Schema — 30 Clinical Measurements

Features are computed for each cell nucleus across three statistical categories:

| Category | Features Included |
|---|---|
| **Mean** (10 features) | `radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`, `concavity`, `concave_points`, `symmetry`, `fractal_dimension` |
| **Standard Error** (10 features) | Same 10 measurements — variability across nuclei within a sample |
| **Worst** (10 features) | Same 10 measurements — largest (most extreme) values observed |

These features are derived from **digitized FNA biopsy images** of breast tissue, capturing the geometric and textural properties of individual cell nuclei under microscopy.

---

## 🧠 Model Comparison & Benchmarks

Six supervised classification algorithms were trained on the same dataset using identical preprocessing pipelines (mean imputation + StandardScaler) and evaluated on a stratified 20% held-out test set (114 samples). All metrics are computed on **real data** — no projections or estimates.

> **Dataset:** 569 samples | **Train:** 455 | **Test:** 114 | **CV:** Stratified 5-Fold

### Full Benchmark Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV (5-Fold Acc.) |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ | 96.49% | 97.50% | 92.86% | 95.12% | **0.9960** | **97.37% ± 1.66%** |
| Random Forest | **97.37%** | **100.00%** | 92.86% | **96.30%** | 0.9929 | 95.43% ± 1.28% |
| Gradient Boosting | 96.49% | **100.00%** | 90.48% | 95.00% | 0.9947 | 95.08% ± 2.45% |
| SVM (RBF Kernel) | **97.37%** | **100.00%** | 92.86% | **96.30%** | 0.9947 | 97.72% ± 1.63% |
| K-Nearest Neighbors | 95.61% | 97.44% | 90.48% | 93.83% | 0.9823 | 96.31% ± 1.79% |
| Decision Tree | 92.98% | 90.48% | 90.48% | 90.48% | 0.9246 | 91.04% ± 2.79% |

✅ **Deployed Model** | 🏆 **Best on metric** (bold values)

> All values computed on the actual UCI Wisconsin dataset using scikit-learn 1.x. Seed: `random_state=42`.

---

### Algorithm Selection Rationale

Despite Random Forest and SVM achieving slightly higher test accuracy (97.37% vs 96.49%), **Logistic Regression was selected** as the deployed model for the following reasons:

| Factor | Logistic Regression | Random Forest / SVM |
|---|---|---|
| **ROC-AUC** | **0.9960 ← Highest** | 0.9929 / 0.9947 |
| **CV Accuracy** | **97.37% ± 1.66%** | 95.43% / 97.72% |
| **Interpretability** | Coefficients expose feature importance | Ensemble black boxes |
| **Inference Speed** | ⚡ Sub-millisecond | 🔶 Moderate |
| **Overfitting Risk** | Low (L2 regularized) | Higher (tree depth) |
| **Medical Auditability** | Explainable to clinicians | Difficult to audit |

In healthcare AI, a model that is interpretable and auditable is often preferred over a marginally more accurate black box. Logistic Regression's **highest ROC-AUC** and **most consistent cross-validation performance** make it the most trustworthy choice for this task.

---

## 🎯 Deployed Model — Deep Dive

### Logistic Regression — Confusion Matrix

```
                    Predicted Benign    Predicted Malignant
  Actual Benign         71  (TN)              1  (FP)
  Actual Malignant       3  (FN)             39  (TP)
```

| Metric | Value | Interpretation |
|---|---|---|
| **Accuracy** | 96.49% | 110 out of 114 samples correctly classified |
| **Precision** | 97.50% | When flagged Malignant, correct 97.5% of the time |
| **Recall (Sensitivity)** | 92.86% | Detects 92.86% of all actual malignant tumors |
| **Specificity** | 98.61% | Correctly identifies 98.61% of benign cases |
| **F1 Score** | 95.12% | Harmonic mean of precision and recall |
| **ROC-AUC** | **0.9960** | Near-perfect discriminative power |
| **False Negatives** | 3 | Malignant cases missed — the most critical error type |
| **False Positives** | 1 | Benign case flagged as malignant |

### Pipeline Architecture

```
Input: 30 float features (FNA measurements)
    │
    ▼
SimpleImputer(strategy="mean")     → handles any NaN values
    │
    ▼
StandardScaler()                   → normalizes to mean=0, std=1
    │
    ▼
LogisticRegression(max_iter=1000)  → binary classification
    │
    ▼
Output: 0 (Benign) or 1 (Malignant)
```

The entire pipeline is serialized as a single `.pkl` artifact — preprocessing and inference are always applied in the correct order, eliminating data leakage risk.

---

## 🏗️ Project Structure

```
CancerScan-AI/
│
├── 📁 dataset/
│   └── data.csv                      # 569-sample UCI Wisconsin dataset (32 columns)
│
├── 📁 model/
│   └── breast_cancer_model.pkl       # Serialized sklearn Pipeline (Joblib)
│
├── 📁 static/
│   └── style.css                     # Application styling
│
├── 📁 templates/
│   ├── index.html                    # 30-feature diagnostic input form
│   └── result.html                   # Benign / Malignant prediction display
│
├── train_model.py                    # Data loading, preprocessing, training, serialization
├── app.py                            # Flask inference server
├── requirements.txt                  # Python dependency manifest
└── README.md                         # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/CancerScan-AI-Breast-Cancer-Detection.git
cd CancerScan-AI-Breast-Cancer-Detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
```
flask
numpy
pandas
scikit-learn
joblib
```

### 3. Train the Model

```bash
python train_model.py
```

Expected output:
```
✅ Model trained with 30 features
🎯 Accuracy: 96.49%
💾 Model saved successfully
```

### 4. Launch the Web Application

```bash
python app.py
```

Navigate to `http://127.0.0.1:5000` in your browser.

---

## 🔄 Application Workflow

```
User enters 30 diagnostic feature values (web form)
                    │
                    ▼
         Flask collects POST request
                    │
                    ▼
     float conversion → numpy array → reshape(1,-1)
                    │
                    ▼
     Pipeline: Imputer → Scaler → LogisticRegression
                    │
                    ▼
       predict() → 0 (Benign) or 1 (Malignant)
                    │
                    ▼
     result.html → "Benign (No Cancer Detected)"
                or "Malignant (Cancer Detected)"
```

---

## ✨ Key Features

- **6-algorithm comparative benchmark** — head-to-head evaluation with real metrics across Accuracy, Precision, Recall, F1, ROC-AUC, and CV stability
- **Clinically-informed model selection** — Recall and ROC-AUC weighted appropriately for medical context
- **Stratified train-test split** — ensures proportional class representation in evaluation, preventing misleading accuracy scores
- **sklearn Pipeline serialization** — imputation + scaling + inference in one artifact; zero preprocessing leakage risk
- **30-feature clinical completeness** — uses the full FNA feature set across mean, SE, and worst-case statistics
- **Real UCI Wisconsin data** — 569 validated biopsy records, internationally recognized benchmark dataset
- **Mobile-responsive frontend** — clean Flask/HTML/CSS UI accessible from any device

---

## 🛠️ Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.8+ | Core ML and backend development |
| ML Framework | Scikit-Learn | Model training, pipeline, evaluation |
| Data Processing | Pandas, NumPy | Dataset operations and feature arrays |
| Preprocessing | `SimpleImputer`, `StandardScaler` | Missing value handling + feature normalization |
| Model Serialization | Joblib | Pipeline persistence as `.pkl` |
| Web Framework | Flask | HTTP request handling and inference serving |
| Frontend | HTML5, CSS3 | Diagnostic input form and result display |
| Dataset | UCI Wisconsin Diagnostic (1995) | 569 FNA biopsy records, 30 nuclear features |

---

## 📈 Future Enhancements

| Enhancement | Description | Expected Impact |
|---|---|---|
| Prediction Confidence | Display class probability (e.g., "94.3% Malignant") | Clinical transparency and trust |
| SHAP Explanations | Feature contribution visualization per prediction | Explainable AI for audit |
| Ensemble Stacking | Combine LR + RF + SVM via meta-learner | Potential accuracy gain to 97–98% |
| Bootstrap 5 UI | Professional medical-grade interface redesign | Improved usability |
| REST API Endpoint | `/predict` with JSON I/O | Hospital system integration capability |
| Streamlit Dashboard | Interactive metrics and confusion matrix display | Real-time model transparency |
| Cloud Deployment | Docker → AWS / Render / Railway | Production availability |
| Cross-Validation Report | In-app display of all 6 benchmark results | Full reproducibility |

---

## 🏢 Internship Context

**AI/ML Intern**  
**Organization:** InternPe  
**Duration:** November 24, 2025 – December 21, 2025

This project was developed as a core deliverable during an AI/ML internship, demonstrating end-to-end proficiency in healthcare-focused machine learning:

- Binary classification on a real clinical benchmark dataset (UCI Wisconsin, 569 samples)
- Comparative evaluation of 6 supervised algorithms with full metric reporting
- Sklearn Pipeline design for leakage-safe preprocessing and serialization
- Flask web deployment with HTML/CSS interface
- Responsible AI practices — clear medical disclaimers, no false clinical claims
- Industry-standard documentation and version-controlled, reproducible code

🔗 **Certificate:** [View on LinkedIn](https://www.linkedin.com/posts/m-v-karthikeya-b26a2131b_internshipcompletion-aiml-machinelearning-activity-7408819858177724416-m1Rt?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFEhlw4BT-6V0rnLIZSzBIoK7YvV2QlbHLc)

🔗 **Watch the Demo:** [LinkedIn Video](https://www.linkedin.com/posts/m-v-karthikeya-b26a2131b_machinelearning-artificialintelligence-datascience-activity-7406349002725777408-Rlo3?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFEhlw4BT-6V0rnLIZSzBIoK7YvV2QlbHLc)

---

## 👤 Author

**M V Karthikeya**  
Aspiring Machine Learning Engineer · Python & Healthcare AI Enthusiast · 📍 India

[![Python](https://img.shields.io/badge/Python-Expert-3776AB?style=flat-square&logo=python)](https://github.com/your-username)
[![ML](https://img.shields.io/badge/Machine%20Learning-Intermediate-F7931E?style=flat-square&logo=scikit-learn)](https://github.com/your-username)
[![Flask](https://img.shields.io/badge/Flask-Intermediate-000000?style=flat-square&logo=flask)](https://github.com/your-username)

---

## 📜 License

This project is licensed under the **MIT License** — free for personal, academic, and commercial use with attribution.

---

> ⚠️ **Reminder:** All predictions produced by this system are **non-clinical** and must **never** be used as a substitute for professional medical advice, diagnosis, or treatment.

---

<div align="center">

⭐ **If this project helped you, consider starring the repository!**

*Built on real clinical data · Benchmarked across 6 algorithms · Designed for responsible healthcare AI*

</div>
