# 📡 IBM Telco Churn Intelligence & Retention AI

An enterprise-grade, interactive Machine Learning & SHAP Explainability platform built for telecommunication companies, retention teams, and customer success leaders to predict and mitigate customer churn.

---

## 🌟 Executive Summary & Business Value

Customer acquisition costs in telecommunications are 5–7x higher than retention costs. This platform delivers:
1. **Accurate Churn Prediction**: Handles class imbalance via SMOTE combined with tuned XGBoost.
2. **Explainable AI (XAI)**: Demystifies the black-box model using SHAP (SHapley Additive exPlanations) so retention teams understand *why* individual accounts leave.
3. **Interactive "What-If" Retention Simulator**: Allows account managers to simulate contract upgrades and service bundles to see immediate risk reduction before offering them to customers.
4. **Batch Customer Scoring**: Uploads customer cohorts, stratifies them into Low/Moderate/High risk tiers, and exports prioritized contact lists for retention campaigns.

---

## 🚀 Key Modules & Interactive Features

### 1. 📊 Executive KPI Dashboard
- **Key Metrics**: Test portfolio size, overall predicted churn rate, count of high-risk accounts (>65% probability), and top churn triggers.
- **Interactive Visualizations**:
  - Churn probability distribution by **Contract Duration** (Month-to-month vs 1-Year vs 2-Year).
  - Churn risk across **Internet Service Type** (Fiber optic vs DSL vs No internet).
  - Risk distribution vs **Tenure Months** (identifies critical customer lifetime drop-off periods).
  - Churn rate broken down by **Payment Method** (Electronic check vs Auto-pay).

### 2. 🧠 Global SHAP Explainability
- **Mean Absolute SHAP Bar Plot**: Ranks the features that have the strongest overall impact on model decisions.
- **Beeswarm Summary Plot**: Highlights feature directionality (e.g., how Month-to-Month contracts push probability into the churn zone).
- **Interactive Dependence Plot**: Select any feature from a dropdown to see its exact non-linear relationship with churn log-odds and automated interaction detection.

### 3. 🔍 Individual Customer Deep-Dive & Waterfall Explanations
- Filter accounts by risk tier (*High Risk Only*, *Moderate*, *Low*).
- Select any customer to view their complete service profile, predicted churn probability, and actual outcome.
- Generates a **Local SHAP Waterfall Plot** showing the exact positive (red) and negative (blue) forces that shaped that customer's prediction.

### 4. ⚡ Interactive Retention "What-If" Simulator
- Configure customer parameters (tenure, contract, internet type, tech support, online security, payment method).
- Live churn probability gauge and risk tier classification.
- **Simulate Retention Offers**:
  - *Offer 1-Year Contract Discount*
  - *Include Free Tech Support & Security*
  - *Switch to Auto-Pay (Credit Card/Bank)*
- Real-time **Save Impact** calculation (*e.g., "Churn probability dropped by 45.2 percentage points!"*).

### 5. 📁 Batch Scoring & CSV Export
- Upload any new `.csv` or `.xlsx` file containing customer accounts.
- The model automatically preprocesses, predicts probabilities, and assigns risk tiers.
- One-click **Download Scored Accounts CSV** button to feed directly into customer relationship management (CRM) systems or call-center dialers.

---

## 🛠️ Tech Stack & Architecture

- **Machine Learning**: `xgboost`, `scikit-learn`, `imblearn` (SMOTE)
- **Model Explainability**: `shap` (TreeExplainer, Waterfall, Beeswarm, Dependence)
- **Interactive Web UI**: `streamlit`
- **Data Visualizations**: `plotly`, `matplotlib`
- **Serialization**: `joblib`

---

## 🏃 How to Run

### Method 1: Double-Click (Windows)
Double-click `run_dashboard.bat` in this folder.

### Method 2: Command Line
Open a terminal in this directory and execute:
```bash
streamlit run app.py
```

The application will launch in your browser at:
`http://localhost:8501`

---

## 💼 Freelance & Client Presentation Tips

When presenting this project to prospective clients or interviewers:
1. **Lead with Business ROI**: Start with the *Executive KPI Dashboard* to demonstrate how much revenue is at risk.
2. **Demonstrate Trust via Explainability**: Show the *SHAP Waterfall Plot* to prove the model provides transparent, auditable business logic rather than arbitrary predictions.
3. **Showcase Actionability with the Simulator**: Demo the *What-If Simulator* live—change a customer from *Month-to-month* to *One year* and show how the risk plummets. This proves your solution directly drives revenue retention.
