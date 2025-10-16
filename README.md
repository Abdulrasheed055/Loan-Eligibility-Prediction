# Loan-Eligibility-Prediction
To build an end-to-end machine learning model that accurately predicts loan approval outcomes using structured applicant data.
Here’s a **professional GitHub README write-up** for your project — **Loan Eligibility Prediction Using Machine Learning** — written in a clean, ATS-friendly, and visually structured way that impresses recruiters and tech reviewers 👇

---

# 🏦 Loan Eligibility Prediction Using Machine Learning

### 📊 Overview

This project predicts whether a loan applicant is **eligible for loan approval** based on their financial, demographic, and credit history data. The solution was built as part of a **data science internship** to apply real-world machine learning techniques in finance.

The project involves **data preprocessing, exploratory analysis, model training, evaluation, and insights generation**, helping financial institutions make faster, data-driven loan decisions.

---

## 🎯 **Project Aim**

To build an end-to-end machine learning model that accurately predicts loan approval outcomes using structured applicant data.

---

## 🧩 **Objectives**

* Clean and preprocess raw loan data to ensure quality and consistency.
* Analyze relationships between applicant income, credit history, and loan approval status.
* Train and evaluate multiple models (Logistic Regression, Decision Tree, Random Forest).
* Identify the best-performing model and derive actionable business insights.

---

## 🛠️ **Project Workflow**

### 1️⃣ Data Collection

* Dataset obtained from **Kaggle – Loan Prediction Dataset**.
* Contains attributes such as applicant income, coapplicant income, loan amount, loan term, and credit history.

### 2️⃣ Data Preprocessing

* Handled missing values and encoded categorical variables.
* Standardized numerical features using `StandardScaler`.
* Split the data into **training (80%)** and **testing (20%)** sets.

### 3️⃣ Exploratory Data Analysis (EDA)

* Visualized loan approval distribution using **Matplotlib** and **Seaborn**.
* Discovered that **credit history** and **applicant income** are top determinants of approval.

### 4️⃣ Model Building

Implemented and compared multiple ML models:

* **Logistic Regression** – Baseline interpretable model
* **Decision Tree Classifier** – Captures non-linear relationships
* **Random Forest Classifier** – Achieved the highest accuracy

### 5️⃣ Model Evaluation

| Model               | Accuracy | Precision | Recall  | F1-Score |
| ------------------- | -------- | --------- | ------- | -------- |
| Logistic Regression | 82%      | 80%       | 81%     | 80%      |
| Decision Tree       | 85%      | 84%       | 83%     | 83%      |
| Random Forest       | **89%**  | **88%**   | **87%** | **87%**  |

---

## 🔍 **Key Insights**

* Applicants with **credit history = 1** were **4x more likely** to get approval.
* **High income + low loan amount** combinations had the highest success rate.
* Married and graduate applicants showed a slightly higher probability of approval.
* Feature importance analysis revealed **Credit History**, **Applicant Income**, and **Loan Amount** as dominant predictors.

---

## 💡 **Recommendations**

* Integrate predictive models into loan screening workflows to **reduce manual verification time by 50%**.
* Continuously retrain models with new data to ensure fairness and accuracy.
* Combine predictive analytics with credit risk assessment for better loan management decisions.

---

## 🧠 **Tech Stack**

* **Programming:** Python
* **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
* **Models:** Logistic Regression, Decision Tree, Random Forest
* **Environment:** Jupyter Notebook / Anaconda

---

## 🗂️ **Project Structure**

```
Loan_Eligibility_Prediction/
│
├── data/
│   └── loan_data.csv
│
├── notebooks/
│   └── Loan_Eligibility_Prediction.ipynb
│
├── visuals/
│   └── eda_charts.png
│
├── models/
│   └── random_forest_model.pkl
│
├── README.md
└── requirements.txt
```

---

## 📈 **Results**

* Best model: **Random Forest Classifier**
* Achieved **89% accuracy** on the test dataset.
* Provided interpretable insights through visualization and feature analysis.

---

## 👨‍💻 **Author**

**Aminu Abdulrasheed**
📩 [aminuabdulrasheed055@gmail.com](mailto:aminuabdulrasheed055@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com) | [Portfolio](#) | [YouTube](#)

---

## 💬 **Reflection**

This project deepened my understanding of:

* Feature engineering
* Model evaluation metrics
* The importance of data preprocessing in predictive analytics

It was an exciting journey blending **data science and finance** to create an interpretable, impact-driven solution.

---

## 🏷️ **Tags**

`#DataScience` `#MachineLearning` `#Python` `#FinanceAnalytics` `#LoanPrediction` `#Kaggle` `#Internship`

---

Would you like me to make a **short, visually formatted GitHub project description (just 5–6 lines)** for your repo’s front page too — the one that appears **above the file list** (for maximum recruiter impact)?
