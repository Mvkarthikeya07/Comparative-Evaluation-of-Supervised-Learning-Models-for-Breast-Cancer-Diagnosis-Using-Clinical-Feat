# 🩺 Machine Learning–Based Breast Cancer Detection and Classification System

**CancerScan AI** is an end-to-end **Machine Learning–powered web application** that predicts whether a breast tumor is **Benign** or **Malignant** using clinically significant diagnostic features.

The system is built using the **Wisconsin Breast Cancer Diagnostic Dataset** and deployed as a **Flask-based web application** with a clean, mobile-friendly user interface.

> ⚠️ **Disclaimer**: This project is developed **strictly for academic and educational purposes only**. It must **not** be used for real-world medical diagnosis or clinical decision-making.

---

## 🚀 Key Features

* 🔍 Predicts **Benign / Malignant** breast tumors
* 🧠 Uses **30 diagnostic features** extracted from cell nucleus images
* 🧪 Robust preprocessing with **missing-value handling**
* ⚙️ Complete ML pipeline (**data → training → inference → deployment**)
* 🌐 Web-based, app-like UI (**mobile-friendly**)
* 📊 High prediction accuracy using **classical Machine Learning** techniques

---

## 🧠 Machine Learning Approach

* **Algorithm**: Logistic Regression
* **Preprocessing**:

  * Mean-based missing value imputation
  * Feature scaling using `StandardScaler`
* **Dataset**: Wisconsin Breast Cancer Diagnostic Dataset
* **Target Variable**: `diagnosis`

  * `M` → Malignant
  * `B` → Benign

**Model Performance**:

* Accuracy: **~96% on test data**

---

## 🧰 Tech Stack

| Layer                | Technology                  |
| -------------------- | --------------------------- |
| Programming Language | Python                      |
| ML Libraries         | Pandas, NumPy, Scikit-learn |
| ML Model             | Logistic Regression         |
| Backend              | Flask                       |
| Frontend             | HTML, CSS                   |
| Deployment           | Local Flask Server          |

---

## 📁 Project Structure

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
├── screenshots/
│   ├── home_page.png
│   ├── benign_result.png
│   └── malignant_result.png
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

The dataset contains **569 patient samples** with **32 columns**, including:

* `id` → Non-predictive (dropped)
* `diagnosis` → Target label
* **30 real-valued diagnostic features**, such as:

  * `radius_mean`
  * `texture_mean`
  * `perimeter_mean`
  * `area_mean`
  * `concavity_worst`
  * `symmetry_worst`
  * `fractal_dimension_worst`

These features represent **quantitative measurements of cell nuclei** extracted from digitized breast tissue images.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/CancerScan-AI-Breast-Cancer-Detection.git
cd CancerScan-AI-Breast-Cancer-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train the Model

```bash
python train_model.py
```

### 4️⃣ Run the Web Application

```bash
python app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

---

## 📸 Application Screenshots

* **Home / Input Page**

<img width="1366" height="768" alt="Screenshot (30)" src="https://github.com/user-attachments/assets/62f23fb7-6eb1-4082-a047-a243191eab22" />

<img width="1366" height="768" alt="Screenshot (27)" src="https://github.com/user-attachments/assets/d8b36f97-ba8a-4426-8c7c-a7f27a0902bb" />

<img width="1366" height="768" alt="Screenshot (28)" src="https://github.com/user-attachments/assets/773123f9-1f46-4480-800b-01a39d4d52de" />

* **Benign Prediction Result**

<img width="1366" height="768" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/ce3bdc92-8621-4ffd-82f9-dbdc098b6e05" />

* **Malignant Prediction Result**

<img width="1366" height="768" alt="Screenshot (31)" src="https://github.com/user-attachments/assets/35ce9a9f-db1e-4e89-b5ff-403f1d7c3aca" />

---

## 🧪 Sample Usage

1. Users enter diagnostic feature values into the web interface
2. The trained ML model processes the inputs
3. The system instantly predicts:

   * ✅ **Benign (No Cancer Detected)**
   * ❌ **Malignant (Cancer Detected)**

---

🏢 Internship Context

This project was developed during my AI/ML Internship at InternPe
(Nov 24, 2025 – Dec 21, 2025).

The work focused on applying practical machine learning concepts in healthcare AI, including:

Medical dataset preprocessing and feature scaling

Supervised classification using Logistic Regression

Model evaluation and performance analysis

Backend development using Flask

Deployment of ML models as web applications

Designing interpretable and user-friendly healthcare ML systems

This project represents academic and practical work completed during the internship period, emphasizing responsible AI development in healthcare applications.

## 🎓 Academic & Educational Value

This project demonstrates:

* Supervised classification using Machine Learning
* Healthcare-focused AI application development
* End-to-end ML system design
* Model deployment using Flask
* Clean UI integration with ML inference

**Ideal for**:

* Final-year academic projects
* Machine Learning internships
* Resume and portfolio showcases
* LinkedIn project demonstrations

---

## 🔮 Future Enhancements

* Display prediction confidence (%)
* Feature importance visualization
* Enhanced UI using Bootstrap
* Streamlit deployment
* Cloud hosting (AWS / Render / Heroku)

---

## 👤 Author

**M V Karthikeya**
Aspiring Machine Learning Engineer | Python & Computer Vision Enthusiast
📍 India

---

##📜 License

This project is licensed under the MIT License.

All predictions are **non-clinical** and must **not** be used for real-world medical decision-making.
