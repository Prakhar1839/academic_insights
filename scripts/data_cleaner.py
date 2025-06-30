import pandas as pd

def clean_students(df):
    df = df.copy()
    df["Intership"] = df["Intership"].replace({"TRUE": 1, "FALSE": 0})
    df["extracurricular_activities"] = df["extracurricular_activities"].replace({"TRUE": 1, "FALSE": 0})
    df["GPA"] = pd.to_numeric(df["GPA"], errors="coerce")
    df["stress_level"] = pd.to_numeric(df["stress_level"], errors="coerce")
    return df.dropna(subset=["Id", "GPA", "Grade"])

def clean_attendance(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Present"] = df["Present"].astype(int)
    return df.dropna(subset=["StudentID", "Subject", "Date"])

def clean_grades(df):
    df = df.copy()
    return df.fillna("F")

def clean_timetable(df):
    df = df.copy()
    df["Day"] = df["Day"].str.strip().str.capitalize()
    return df