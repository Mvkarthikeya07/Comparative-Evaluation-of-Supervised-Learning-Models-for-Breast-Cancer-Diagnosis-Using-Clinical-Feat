# 🩺 CancerScan AI — Comparative Evaluation of Supervised Learning Models for Breast Cancer Diagnosis Using Clinical Features

> **End-to-end Machine Learning system for breast cancer diagnosis using clinical diagnostic features.**
>
> Built with **Python · Scikit-learn · Flask · Pandas · NumPy**
>
> Based on the **Wisconsin Breast Cancer Diagnostic Dataset (569 patient records)**

---

# 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Live Demo & Screenshots](#-live-demo--screenshots)
3. [Dataset](#-dataset)
4. [Machine Learning Pipeline](#-machine-learning-pipeline)
5. [Model Performance](#-model-performance)
6. [Project Structure](#-project-structure)
7. [Getting Started](#-getting-started)
8. [Application Workflow](#-application-workflow)
9. [Technologies Used](#-technologies-used)
10. [Future Roadmap](#-future-roadmap)
11. [Internship Background](#-internship-background)
12. [Academic Value](#-academic-value)
13. [Author](#-author)
14. [License](#-license)

---

## 🎯 Project Overview

Breast cancer remains one of the most prevalent cancers worldwide, making early diagnosis critical for improving treatment outcomes and patient survival.

**CancerScan AI** is an end-to-end Machine Learning application that predicts whether a breast tumor is:

* ✅ **Benign (Non-Cancerous)**
* ❌ **Malignant (Cancerous)**

The system utilizes clinically relevant measurements extracted from digitized images of breast tissue cell nuclei and applies supervised learning techniques to perform accurate classification.

Unlike rule-based systems, the model learns diagnostic patterns directly from historical medical data and generalizes them to previously unseen patient samples.

> ⚠️ **Disclaimer**
>
> This project is developed strictly for academic and educational purposes. It is not intended for clinical diagnosis, treatment planning, or medical decision-making.

---

### Core Objectives

| Goal                              | Approach                |
| --------------------------------- | ----------------------- |
| Early cancer risk assessment      | Binary Classification   |
| Learn from clinical features      | Logistic Regression     |
| Handle incomplete medical records | Mean Value Imputation   |
| Improve prediction stability      | StandardScaler          |
| Real-time predictions             | Flask Web Application   |
| Reproducible ML workflow          | Modular Pipeline Design |

---

### Problem Statement

**Type:** Supervised Learning — Binary Classification

**Target Variable:** `diagnosis`

| Value | Meaning   |
| ----- | --------- |
| M     | Malignant |
| B     | Benign    |

The objective is to classify breast tumors based on 30 diagnostic measurements obtained from digitized breast tissue images.

---

## 🖥️ Live Demo & Screenshots

### 🔹 Home Page — Clinical Feature Input Interface

Users enter diagnostic measurements through a clean and responsive web interface.

![Home Page](https://github.com/user-attachments/assets/62f23fb7-6eb1-4082-a047-a243191eab22)

![Input Form](https://github.com/user-attachments/assets/d8b36f97-ba8a-4426-8c7c-a7f27a0902bb)

![Extended Feature Form](https://github.com/user-attachments/assets/773123f9-1f46-4480-800b-01a39d4d52de)

---

### 🔹 Benign Prediction Result

When the model predicts a non-cancerous tumor.

![Benign Prediction](https://github.com/user-attachments/assets/ce3bdc92-8621-4ffd-82f9-dbdc098b6e05)

---

### 🔹 Malignant Prediction Result

When the model predicts a potentially cancerous tumor.

![Malignant Prediction](https://github.com/user-attachments/assets/35ce9a9f-db1e-4e89-b5ff-403f1d7c3aca)

---

### 🔹 End-to-End Workflow

```text
Patient Diagnostic Features
            │
            ▼
      User Input Form
            │
            ▼
      Data Validation
            │
            ▼
 Missing Value Imputation
            │
            ▼
     Feature Scaling
            │
            ▼
 Logistic Regression Model
            │
            ▼
 Benign / Malignant Prediction
            │
            ▼
      Result Display
```

---

## 📊 Dataset

The project uses the **Wisconsin Breast Cancer Diagnostic Dataset**, a widely used benchmark dataset in medical machine learning research.

### Dataset Overview

| Attribute           | Value                 |
| ------------------- | --------------------- |
| Total Samples       | 569                   |
| Predictive Features | 30                    |
| Classes             | Benign, Malignant     |
| Problem Type        | Binary Classification |
| Domain              | Healthcare AI         |

---

### Feature Description

The dataset contains numerical measurements extracted from digitized images of breast tissue cell nuclei.

Examples include:

* radius_mean
* texture_mean
* perimeter_mean
* area_mean
* smoothness_mean
* compactness_mean
* concavity_mean
* concave_points_mean
* symmetry_mean
* fractal_dimension_mean

Additional standard error and worst-case measurements increase the total predictive feature count to **30 clinical attributes**.

---

### Class Distribution

| Diagnosis | Count |
| --------- | ----- |
| Benign    | 357   |
| Malignant | 212   |

This distribution provides a realistic healthcare classification problem while maintaining sufficient representation of both classes.

---

## 🧠 Machine Learning Pipeline

```text
Raw Dataset (data.csv)
          │
          ▼
 Remove Non-Predictive Columns
 (ID Column Removal)
          │
          ▼
 Target Encoding
 M → 1
 B → 0
          │
          ▼
 Train/Test Split (80/20)
          │
          ▼
 Missing Value Imputation
 (Mean Strategy)
          │
          ▼
 StandardScaler
          │
          ▼
 Logistic Regression
          │
          ▼
 Model Serialization
 (Joblib)
          │
          ▼
 Flask Deployment
```

---

### Training Configuration

| Parameter       | Value               |
| --------------- | ------------------- |
| Algorithm       | Logistic Regression |
| Train Split     | 80%                 |
| Test Split      | 20%                 |
| Random State    | 42                  |
| Missing Values  | Mean Imputation     |
| Feature Scaling | StandardScaler      |
| Model Storage   | Joblib              |

---

## 📈 Model Performance

The Logistic Regression model achieved approximately:

| Metric              | Value               |
| ------------------- | ------------------- |
| Accuracy            | ~96%                |
| Classification Type | Binary              |
| Model Type          | Logistic Regression |
| Deployment Status   | Flask Integrated    |

---

### Performance Insights

* High classification accuracy on unseen test samples
* Effective separation between benign and malignant tumors
* Scaled features improve optimization and convergence
* Pipeline-based preprocessing ensures reproducibility

---

### 📉 Model Evaluation Summary

```text
Classification Performance
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Accuracy      ████████████████████  ~96%

Precision     High

Recall        High

F1-Score      High

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

> Future versions can include Decision Trees, Random Forests, Support Vector Machines, and Gradient Boosting for comparative benchmarking.

---

## 🗂️ Project Structure

```text
CancerScan-AI/
│
├── dataset/
│   └── data.csv
│
├── model/
│   └── breast_cancer_model_30.pkl
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

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* pip

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/CancerScan-AI.git

cd CancerScan-AI
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the Model

```bash
python train_model.py
```

Expected Output:

```text
Model trained successfully
Accuracy: ~96%
Model saved successfully
```

---

### 4️⃣ Run the Web Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## ⚙️ Application Workflow

```text
Step 1 → User enters diagnostic measurements

Step 2 → Flask receives POST request

Step 3 → Missing values handled automatically

Step 4 → Data standardized using StandardScaler

Step 5 → Logistic Regression predicts diagnosis

Step 6 → Result displayed as:

         ✅ Benign

         OR

         ❌ Malignant
```

---

## 🛠️ Technologies Used

| Layer                  | Technology          |
| ---------------------- | ------------------- |
| Programming Language   | Python              |
| Data Processing        | Pandas, NumPy       |
| Machine Learning       | Scikit-learn        |
| Classification Model   | Logistic Regression |
| Missing Value Handling | SimpleImputer       |
| Feature Scaling        | StandardScaler      |
| Model Persistence      | Joblib              |
| Backend Framework      | Flask               |
| Frontend               | HTML5, CSS3         |

---

## 🔮 Future Roadmap

| Enhancement                       | Expected Impact |
| --------------------------------- | --------------- |
| Random Forest Benchmarking        | High            |
| Support Vector Machine Comparison | High            |
| Decision Tree Evaluation          | High            |
| Prediction Confidence Scores      | High            |
| Feature Importance Visualization  | Medium          |
| Explainable AI (SHAP/LIME)        | High            |
| Streamlit Deployment              | Medium          |
| Cloud Deployment (AWS/Azure/GCP)  | Medium          |
| Interactive Analytics Dashboard   | Medium          |

---

## 🏢 Internship Background

### AI/ML Intern — InternPe

**Duration:** November 24, 2025 – December 21, 2025

This project reflects practical experience gained during an AI/ML internship, including:

* Medical dataset preprocessing
* Supervised classification techniques
* Feature scaling and normalization
* Logistic Regression implementation
* Machine Learning model evaluation
* Flask-based deployment
* Healthcare-focused AI system development

The project demonstrates the application of machine learning concepts to a real-world healthcare classification problem.

---

## 🎓 Academic Value

This project demonstrates:

* Binary Classification using Machine Learning
* Healthcare AI Applications
* Clinical Data Analysis
* Data Preprocessing Pipelines
* Model Deployment with Flask
* End-to-End ML System Development

### Suitable For

* Academic Mini Projects
* Final-Year Projects
* Machine Learning Portfolios
* Internship Applications
* LinkedIn Project Showcases

---

## 👤 Author

**M V Karthikeya**

Machine Learning Enthusiast • Python Developer • Healthcare AI Projects

---

## 📜 License

This project is licensed under the **MIT License**.

All predictions generated by this application are strictly for educational and demonstration purposes and must not be used for real-world medical diagnosis.

---

> ⭐ If you found this project useful, consider starring the repository.
>
> Contributions, improvements, and healthcare AI discussions are welcome.
