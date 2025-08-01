# Academic Insights Dashboard
# 📊 Academic Insights Dashboard

A Streamlit-powered dashboard that visualizes student performance, tracks academic health, and uses machine learning to predict student risk levels — all in real time.

## 🚀 Features

- 🎓 Individual student profiles with performance metrics
- 📈 GPA, attendance, and stress-level analysis
- 🤖 ML-powered risk prediction (On Track vs At Risk)
- 📥 Downloadable student reports (CSV)
- 🔎 Department-level filtering and heatmaps
- 📅 Timetable and subject-wise breakdowns

## 🧠 Machine Learning

Model: **Random Forest Classifier**  
Trained on:
- GPA
- Attendance %
- Study/sleep hours
- Stress levels
- Extracurriculars & internships  
Output: `academic_risk` → `"At Risk"` or `"On Track"`

## 📂 Folder Structure

This Streamlit app analyzes student performance with ML and visual insights.

## 🔧 How to Run
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
## train the model
'''bash
python train_model.py
'''
## how to run
'''bash
streamlit run streamlit_app.py
'''

Made by Prakhar 💡#   a c a d e m i c _ i n s i g h t s 
 
 
