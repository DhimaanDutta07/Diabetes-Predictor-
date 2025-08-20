from flask import Flask, render_template, request
import pandas as pd
import joblib
from collections import deque

app = Flask(__name__)
model = joblib.load("health_risk_model.pkl")
history = deque(maxlen=5)

def risk_category(prob):
    if prob < 30:
        return "Low Risk", "#2ecc71", "Maintain a healthy lifestyle and regular checkups."
    elif prob < 70:
        return "Medium Risk", "#f1c40f", "Moderate risk: watch diet, exercise regularly."
    else:
        return "High Risk", "#e74c3c", "High risk: consult a doctor immediately."

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    risk_prob = None
    risk_level = None
    color = "#2ecc71"
    advice = ""
    if request.method == "POST":
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        blood_pressure = float(request.form["blood_pressure"])
        glucose = float(request.form["glucose"])
        insulin = float(request.form["insulin"])
        skin_thickness = float(request.form["skin_thickness"])
        pregnancies = float(request.form["pregnancies"])
        diabetes_pedigree = float(request.form["diabetes_pedigree"])

        input_df = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness, insulin,
            bmi, diabetes_pedigree, age
        ]], columns=["Pregnancies","Glucose","BloodPressure","SkinThickness",
                     "Insulin","BMI","DiabetesPedigreeFunction","Age"])

        prob = model.predict_proba(input_df)[0][1] * 100
        risk_prob = round(prob, 2)
        risk_level, color, advice = risk_category(risk_prob)
        prediction = "Positive" if prob >= 50 else "Negative"

        history.appendleft({
            "risk_level": risk_level,
            "risk_prob": risk_prob,
            "advice": advice
        })

    return render_template("index.html", prediction=prediction,
                           risk_prob=risk_prob, risk_level=risk_level,
                           color=color, advice=advice, history=list(history))

if __name__ == "__main__":
    app.run(debug=True)
