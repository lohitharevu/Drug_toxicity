from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")
imputer = joblib.load("imputer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        logP = float(data["logP"])
        qed = float(data["qed"])
        SAS = float(data["SAS"])

        features = np.array([[logP, qed, SAS]])
        features = imputer.transform(features)

        prob = model.predict_proba(features)[0][1]
        confidence = round(prob * 100, 2)

        if confidence < 20:
            risk = "🟢 Safe"
        elif confidence < 50:
            risk = "🟡 Low Risk"
        elif confidence < 75:
            risk = "🟠 Moderate"
        else:
            risk = "🔴 High Risk"

        prediction = "Harmful ⚠️" if prob > 0.5 else "Safe ✅"

        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "risk": risk
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
