# 🩺 CancerScan AI — Breast Cancer Diagnosis Using Machine Learning

> **End-to-end Machine Learning system for breast cancer risk classification using clinical diagnostic features.**
> Built with Python · Scikit-learn · Flask · Pandas | Wisconsin Breast Cancer Diagnostic Dataset

---

# 📋 Table of Contents

1. Project Overview
2. Live Demo & Screenshots
3. Dataset
4. Machine Learning Pipeline
5. Model Performance
6. Project Structure
7. Getting Started
8. Application Workflow
9. Technologies Used
10. Future Roadmap
11. Internship Background
12. Author
13. License

---

# 🎯 Project Overview

Breast cancer is among the most common forms of cancer worldwide. Early detection plays a crucial role in improving treatment outcomes and patient survival rates.

CancerScan AI is an end-to-end Machine Learning application that predicts whether a breast tumor is:

* ✅ Benign (Non-Cancerous)
* ❌ Malignant (Cancerous)

The system leverages clinically relevant diagnostic measurements extracted from digitized breast tissue images and provides instant predictions through an intuitive Flask-based web interface.

> ⚠️ Disclaimer:
> This project is developed strictly for educational and academic purposes and must not be used for real-world medical diagnosis or clinical decision-making.

---

## Core Objectives

| Goal                                    | Approach                    |
| --------------------------------------- | --------------------------- |
| Early breast cancer risk assessment     | Supervised Classification   |
| Learn from clinical diagnostic features | Logistic Regression         |
| Handle incomplete records               | Mean-value imputation       |
| Improve model stability                 | Feature scaling             |
| Deliver real-time predictions           | Flask web application       |
| Maintain reproducible ML workflow       | Pipeline-based architecture |

---

## Problem Statement

**Type:** Supervised Learning — Binary Classification

**Target Variable:** `diagnosis`

* M → Malignant
* B → Benign

The model learns patterns from historical clinical measurements and predicts tumor classification for previously unseen samples.

---

# 🖥️ Live Demo & Screenshots

## 🔹 Home Page — Clinical Feature Input Interface

Users enter diagnostic measurements obtained from breast tissue analysis.

![Home Page](https://github.com/user-attachments/assets/62f23fb7-6eb1-4082-a047-a243191eab22)

![Input Form](https://github.com/user-attachments/assets/d8b36f97-ba8a-4426-8c7c-a7f27a0902bb)

![Extended Feature Form](https://github.com/user-attachments/assets/773123f9-1f46-4480-800b-01a39d4d52de)

---

## 🔹 Benign Prediction Result

The model predicts a non-cancerous tumor classification.

![Benign Result](https://github.com/user-attachments/assets/ce3bdc92-8621-4ffd-82f9-dbdc098b6e05)

---

## 🔹 Malignant Prediction Result

The model predicts a potentially cancerous tumor classification.

![Malignant Result](https://github.com/user-attachments/assets/35ce9a9f-db1e-4e89-b5ff-403f1d7c3aca)

---

## 🔹 End-to-End Workflow

```text
Clinical Measurements
          │
          ▼
 User Input Form
          │
          ▼
 Data Validation
          │
          ▼
 Missing Value Handling
          │
          ▼
 Feature Scaling
          │
          ▼
 Logistic Regression Prediction
          │
          ▼
 Benign / Malignant Classification
```

---

# 📊 Dataset

The project uses the Wisconsin Breast Cancer Diagnostic Dataset, one of the most widely used benchmark datasets in medical machine learning research.

## Dataset Overview

| Attribute     | Value                   |
| ------------- | ----------------------- |
| Total Records | 569                     |
| Features      | 30 Diagnostic Features  |
| Classes       | Benign, Malignant       |
| Problem Type  | Binary Classification   |
| Domain        | Healthcare / Medical AI |

---

## Feature Categories

The dataset contains measurements computed from digitized images of breast mass cell nuclei.

Examples include:

* radius_mean
* texture_mean
* perimeter_mean
* area_mean
* smoothness_mean
* compactness_mean
* concavity_mean
* symmetry_mean
* fractal_dimension_mean

Additional standard error and worst-case measurements are also included, resulting in 30 predictive features.

---

# 🧠 Machine Learning Pipeline

```text
Raw Dataset (data.csv)
          │
          ▼
 Data Cleaning
 (Drop ID columns)
          │
          ▼
 Target Encoding
 M → 1
 B → 0
          │
          ▼
 Train/Test Split
 80% / 20%
          │
          ▼
 Mean Value Imputation
          │
          ▼
 StandardScaler
          │
          ▼
 Logistic Regression
          │
          ▼
 Model Serialization
 breast_cancer_model.pkl
          │
          ▼
 Flask Deployment
```

---

## Training Details

* Algorithm: Logistic Regression
* Missing Value Strategy: Mean Imputation
* Feature Scaling: StandardScaler
* Train/Test Split: 80% / 20%
* Random State: 42
* Stratified Sampling: Enabled
* Model Storage: Joblib

---

# 📈 Model Performance

The trained Logistic Regression model achieves:

| Metric              | Value               |
| ------------------- | ------------------- |
| Accuracy            | ~96%                |
| Classification Type | Binary              |
| Training Method     | Supervised Learning |
| Deployment Status   | Flask Integrated    |

### Performance Interpretation

* High classification accuracy on unseen data
* Balanced handling of benign and malignant classes
* Scaled features improve optimization stability
* Pipeline architecture ensures reproducibility

---

# 🗂️ Project Structure

```text
CancerScan-AI/
│
├── dataset/
│   └── data.csv
│
├── model/
│   └── breast_cancer_model.pkl
│
├── static/
│   └── style.css
│
├── templates/
│   ├── index.html
│   └── result.html
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## Prerequisites

* Python 3.8+
* pip

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/CancerScan-AI.git
cd CancerScan-AI
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model

```bash
python train_model.py
```

Expected Output:

```text
✅ Model trained with 30 features
🎯 Accuracy: ~96%
💾 Model saved successfully
```

### 4️⃣ Launch Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# ⚙️ Application Workflow

```text
Step 1 → User enters 30 diagnostic values

Step 2 → Flask receives POST request

Step 3 → Missing values handled automatically

Step 4 → Features scaled using StandardScaler

Step 5 → Logistic Regression predicts class

Step 6 → Result displayed as:

         ✅ Benign

         OR

         ❌ Malignant
```

---

# 🛠️ Technologies Used

| Layer                  | Technology          |
| ---------------------- | ------------------- |
| Language               | Python              |
| Data Processing        | Pandas, NumPy       |
| Machine Learning       | Scikit-learn        |
| Classification Model   | Logistic Regression |
| Feature Scaling        | StandardScaler      |
| Missing Value Handling | SimpleImputer       |
| Model Persistence      | Joblib              |
| Backend                | Flask               |
| Frontend               | HTML5, CSS3         |

---

# 🔮 Future Roadmap

| Enhancement                         | Impact |
| ----------------------------------- | ------ |
| Prediction Confidence Scores        | High   |
| Random Forest Comparison            | High   |
| Support Vector Machine Benchmarking | High   |
| Feature Importance Visualization    | Medium |
| Interactive Dashboard               | Medium |
| Streamlit Deployment                | Medium |
| Cloud Deployment (AWS/Azure/GCP)    | Medium |
| Explainable AI (SHAP/LIME)          | High   |

---

# 🏢 Internship Background

**AI/ML Intern**
**Organization:** InternPe
**Duration:** Nov 24, 2025 – Dec 21, 2025

This project reflects practical industry exposure in:

* Medical dataset preprocessing
* Healthcare AI applications
* Logistic Regression modeling
* Feature scaling and data preparation
* Machine Learning deployment using Flask
* Building interpretable healthcare-focused ML systems

---

# 👤 Author

**M V Karthikeya**

Machine Learning Enthusiast • Python Developer • Healthcare AI Projects

---

# 📜 License

Licensed under the MIT License.

---

> ⭐ If you found this project useful, consider starring the repository.
>
> Contributions, improvements, and healthcare AI discussions are welcome.
