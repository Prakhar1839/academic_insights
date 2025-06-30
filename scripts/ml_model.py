import pandas as pd
import joblib

model = joblib.load("model/risk_predictor.pkl")

def predict_risk(row: pd.DataFrame) -> str:
    row = row.copy()
    row["Intership"] = row["Intership"].map({"TRUE": 1, "FALSE": 0})
    row["extracurricular_activities"] = row["extracurricular_activities"].map({"TRUE": 1, "FALSE": 0})
    X = row[["GPA", "Attendance (%)", "Study Hours", "Sleep Hours", "stress_level", "Intership", "extracurricular_activities"]]
    return "At Risk" if model.predict(X)[0] == 1 else "On Track"
