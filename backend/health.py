from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("health_risk_model.pkl")

def risk_category(prob):
    if prob < 30:
        return "Low Risk", "#2ecc71", "Maintain a healthy lifestyle and regular checkups."
    elif prob < 70:
        return "Medium Risk", "#f1c40f", "Moderate risk: watch diet, exercise regularly."
    else:
        return "High Risk", "#e74c3c", "High risk: consult a doctor immediately."

@app.route('/', methods=['GET'])
def home():
    return "Health Risk API is ready."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = float(data['age'])
    bmi = float(data['bmi'])
    blood_pressure = float(data['blood_pressure'])
    glucose = float(data['glucose'])
    insulin = float(data['insulin'])
    skin_thickness = float(data['skin_thickness'])
    pregnancies = float(data['pregnancies'])
    diabetes_pedigree = float(data['diabetes_pedigree'])

    input_df = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness, insulin,
        bmi, diabetes_pedigree, age
    ]], columns=["Pregnancies","Glucose","BloodPressure","SkinThickness",
                 "Insulin","BMI","DiabetesPedigreeFunction","Age"])

    prob = model.predict_proba(input_df)[0][1] * 100
    risk_prob = round(prob, 2)
    risk_level, color, advice = risk_category(risk_prob)
    prediction = "Positive" if prob >= 50 else "Negative"

    return jsonify({
        "prediction": prediction,
        "risk_prob": risk_prob,
        "risk_level": risk_level,
        "color": color,
        "advice": advice
    })

if __name__ == "__main__":
    app.run(debug=True)