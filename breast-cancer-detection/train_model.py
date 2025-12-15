import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset/data.csv")

# Drop ID & unnamed columns
data = data.drop(columns=["id"], errors="ignore")
data = data.loc[:, ~data.columns.str.contains("Unnamed")]

# Encode target
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# All 30 features
FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

X = data[FEATURES]
y = data["diagnosis"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline (expects 30 features)
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
accuracy = accuracy_score(y_test, pipeline.predict(X_test))
print(f"✅ Model trained with 30 features")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%")

# Save
joblib.dump(pipeline, "model/breast_cancer_model.pkl")
print("💾 Model saved successfully")
