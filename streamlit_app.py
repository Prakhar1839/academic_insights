import streamlit as st
from scripts.ml_model import predict_risk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts import data_cleaner as dc

st.set_page_config(page_title="Academic Insights", layout="wide")
st.title("📊 Academic Insights Dashboard")

# Load Data
students = pd.read_csv("data/students.csv")
grades = pd.read_csv("data/grades.csv")
attendance = pd.read_csv("data/attendance.csv")
timetable = pd.read_csv("data/timetable.csv")

# Clean Data
students = dc.clean_students(students)
grades = dc.clean_grades(grades)
attendance = dc.clean_attendance(attendance)
timetable = dc.clean_timetable(timetable)


# Filter by department
st.sidebar.header("Filter by Department")
dept = st.sidebar.selectbox("Department", ["All"] + sorted(students["Department"].unique()))
if dept != "All":
    students = students[students["Department"] == dept]

st.subheader("🧾 Student Overview")
st.dataframe(students[["Id", "Name", "GPA", "Attendance (%)", "Grade"]], use_container_width=True)

# GPA Histogram
st.subheader("🎯 GPA Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(students["GPA"], bins=10, color="skyblue", kde=True, ax=ax1)
st.pyplot(fig1)

# Stress boxplot
st.subheader("🧠 Stress Level Overview")
fig2, ax2 = plt.subplots()
sns.boxplot(y=students["stress_level"], ax=ax2, color="salmon")
st.pyplot(fig2)
st.subheader("🚨 ML Risk Prediction")

student_id = st.selectbox("Select Student ID for ML Prediction", students["Id"])
student_row = students[students["Id"] == student_id]

if not student_row.empty:
    prediction = predict_risk(student_row)
    if prediction == "At Risk":
        st.error(f"⚠️ Predicted Status: {prediction}")
    else:
        st.success(f"✅ Predicted Status: {prediction}")
        csv = students.to_csv(index=False).encode("utf-8")
# 📌 Individual Student Report
st.subheader("📌 Individual Student Report")

# Step 1: Select a student by ID
selected_id = st.selectbox("Select Student ID", students["Id"])

# Step 2: Filter that one student's row
student_data = students[students["Id"] == selected_id]

# Step 3: Show personal info and metrics
if not student_data.empty:
    st.markdown(f"### 📄 Report for: {student_data.iloc[0]['Name']}")
    st.write("**Department:**", student_data.iloc[0]["Department"])
    st.write("**GPA:**", student_data.iloc[0]["GPA"])
    st.write("**Attendance (%):**", student_data.iloc[0]["Attendance (%)"])
    st.write("**Stress Level:**", student_data.iloc[0]["stress_level"])
    st.write("**Grade:**", student_data.iloc[0]["Grade"])

    # Optional: show subject-wise grades
    student_grades = grades[grades["StudentID"] == selected_id].drop(columns=["StudentID"])
    if not student_grades.empty:
        st.write("**Subject-wise Grades:**")
        st.dataframe(student_grades.T.rename(columns={student_grades.index[0]: "Grade"}))

    # Step 4: Prepare CSV bytes for download
    csv_data = student_data.to_csv(index=False).encode("utf-8")

    # Step 5: Download button
    st.download_button(
        label="📥 Download This Student's Report (CSV)",
        data=csv_data,
        file_name=f"student_{selected_id}_report.csv",
        mime="text/csv"
    )

# Grade Distribution Bar Chart
st.subheader("📊 Average Grade by Subject")
grade_map = {"A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "C": 2.0, "D": 1.0, "F": 0.0}
grades_numeric = grades.drop(columns=["StudentID"]).applymap(lambda g: grade_map.get(g, 0))
avg_grades = grades_numeric.mean().sort_values()

fig3, ax3 = plt.subplots()
sns.barplot(x=avg_grades.values, y=avg_grades.index, palette="viridis", ax=ax3)
ax3.set_xlabel("Average Score")
st.pyplot(fig3)

# Attendance Heatmap
st.subheader("🌡️ Attendance Heatmap (Subject vs Day)")
attendance = attendance.merge(timetable, on="Subject", how="left")
heatmap_data = attendance.pivot_table(index="Subject", columns="Day", values="Present", aggfunc="mean")
fig4, ax4 = plt.subplots(figsize=(10,6))
sns.heatmap(heatmap_data, annot=True, cmap="YlOrBr", ax=ax4, fmt=".2f")
st.pyplot(fig4)

st.markdown("---")
