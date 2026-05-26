from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# ============================================================
# Load trained model
# ============================================================

with open("airbnb_xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("airbnb_features.pkl", "rb") as f:
    feature_names = pickle.load(f)

# ============================================================
# Home route
# ============================================================

@app.route("/")
def home():

    return {
        "message": "Airbnb price prediction API is running."
    }

# ============================================================
# Prediction route
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Get JSON input
    data = request.get_json()

    # Convert input into DataFrame
    input_data = pd.DataFrame(
        [data],
        columns=feature_names
    )

    # Predict log price
    log_prediction = model.predict(input_data)[0]

    # Convert back to original price
    predicted_price = np.exp(log_prediction)

    # Return prediction
    return jsonify({
        "predicted_log_price": round(float(log_prediction), 3),
        "predicted_price": round(float(predicted_price), 2)
    })

# ============================================================
# Run app
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)