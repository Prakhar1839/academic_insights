import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

df = pd.read_csv("data/students.csv")

df["Intership"] = df["Intership"].replace({"TRUE": 1, "FALSE": 0})
df["extracurricular_activities"] = df["extracurricular_activities"].replace({"TRUE": 1, "FALSE": 0})
df["Risk"] = df["Grade"].apply(lambda g: 1 if g in ["CC", "Fail"] else 0)

X = df[["GPA", "Attendance (%)", "Study Hours", "Sleep Hours", "stress_level", "Intership", "extracurricular_activities"]]
y = df["Risk"]

model = RandomForestClassifier()
model.fit(X, y)

dump(model, "model/risk_predictor.pkl")
print("✅ Model trained and saved successfully.")