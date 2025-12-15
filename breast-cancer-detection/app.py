from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model/breast_cancer_model.pkl"
model = joblib.load(MODEL_PATH)

FEATURES = [
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave points_mean","symmetry_mean","fractal_dimension_mean",
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave points_se","symmetry_se","fractal_dimension_se",
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave points_worst","symmetry_worst","fractal_dimension_worst"
]

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(request.form[f]) for f in FEATURES]
    prediction = model.predict(np.array(values).reshape(1, -1))[0]

    result = "Malignant (Cancer Detected)" if prediction == 1 else "Benign (No Cancer Detected)"
    return render_template("result.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
