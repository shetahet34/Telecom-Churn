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

### 1. 📊 Executive Overview & Revenue Exposure
- **Business KPIs**: Active monitored accounts, predicted churn percentage, high-risk account count (>65% churn chance), and real-time **Monthly & Annual Revenue at Risk** estimation.
- **Visual Hotspots**:
  - Churn percentage distribution by **Contract Duration** (Month-to-month vs 1-Year vs 2-Year).
  - Churn risk across **Internet Service Type** (Fiber optic vs DSL vs No internet).
  - **Tenure Cohort Risk** (identifies critical 0–6 month onboarding vulnerabilities).
  - Risk distribution by **Payment Method** (Electronic check vs Auto-pay).
- **Executive Takeaway Card**: Distills findings into plain-English business recommendations.

### 2. 💡 Why Customers Leave (Plain-English AI Drivers)
- **Top Factor Ranking**: Visualizes the top 10 churn drivers with human-friendly labels and clear color-coding (🔴 Increases Risk vs 🟢 Builds Loyalty).
- **Interactive Factor Explorer & Playbook**: Select any driver (e.g. Contract, Tech Support, Payment Method) to view comparative churn rates alongside an **Executive Playbook** detailing root cause, marketing campaigns, and expected retention boost.

### 3. 👤 Customer Risk Profiler & Call Assistant
- Designed specifically for customer success and retention call center agents.
- **Search & Filter**: Find accounts by Customer ID, City, or Risk Tier (High / Moderate / Low).
- **Local Factor Contribution Chart**: An intuitive horizontal impact chart showing exactly which factors push this specific customer toward leaving or staying.
- **Automated Retention Action Plan**: Dynamically generates tailored next steps for the agent to save the account during renewal calls.

### 4. 🎯 Retention Offer Simulator (What-If Tester)
- **Live Churn Chance Speedometer**: Dynamic Plotly gauge indicating real-time churn chance across green, yellow, and red zones.
- **One-Click Retention Interventions**:
  - *🎁 Upgrade to 1-Year Contract Discount*
  - *🛡️ Bundle Free Tech Support & Online Security*
  - *💳 Migrate to Automated Payment Method*
- **Instant ROI & Revenue Saved**: Displays live before-and-after churn percentage / churn chance and projected annual revenue saved per account.

### 5. 📑 Prioritized Call List & File Scorer
- **Call Queue**: Prioritized list of high-risk customers with real-time search, progress bars, and one-click **Download CSV**.
- **Instant Batch File Scorer**: Drag-and-drop any new `.csv` or `.xlsx` customer list to score all accounts, classify risk tiers, and export enriched reports.

