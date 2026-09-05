import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Intelligence & Retention AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Executive CSS
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Metrics card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e2530 0%, #151a21 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 4px;
    }
    .metric-delta {
        font-size: 0.85rem;
        font-weight: 500;
    }
    .delta-positive { color: #ef4444; }
    .delta-negative { color: #10b981; }

    /* Risk Badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fcd34d;
        border: 1px solid #f59e0b;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.2);
        color: #6ee7b7;
        border: 1px solid #10b981;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Model Assets & Data (Cached)
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    assets = joblib.load("churn_model_assets.joblib")
    explainer = shap.TreeExplainer(assets['model'])
    assets['explainer'] = explainer
    return assets

assets = load_assets()
pipeline = assets['pipeline']
preprocessor = assets['preprocessor']
model = assets['model']
clean_feature_names = assets['clean_feature_names']
categorical_options = assets['categorical_options']
numerical_ranges = assets['numerical_ranges']
x_test = assets['x_test']
y_test = assets['y_test']
x_test_transformed = assets['x_test_transformed']
test_df_enriched = assets['test_df_enriched']
shap_values = assets['shap_values']
explainer = assets['explainer']

# ---------------------------------------------------------
# Sidebar Navigation & Global Info
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3067/3067260.png", width=60)
    st.title("Telco Churn AI")
    st.caption("Executive Retention & Explainability Engine")
    
    st.markdown("---")
    st.subheader("Model Status")
    st.markdown("🟢 **Model**: XGBoost (SMOTE-Balanced)")
    st.markdown(f"📊 **Test Sample Size**: {len(test_df_enriched):,} customers")
    
    avg_churn = (test_df_enriched['Predicted Churn'].mean()) * 100
    st.markdown(f"⚠️ **Predicted Churn**: {avg_churn:.1f}%")
    
    st.markdown("---")
    st.subheader("Decision Threshold")
    threshold = st.slider("Classification Threshold", min_value=0.2, max_value=0.8, value=0.5, step=0.05)
    st.caption("Adjust probability cut-off for classifying a customer as 'High Churn Risk'.")

    st.markdown("---")
    st.markdown("""
    **Business Value**:
    - Reduce customer attrition
    - Pinpoint exact churn triggers
    - Simulate retention offers
    """)

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.title("📡 IBM Telco Churn Intelligence & Explainability Platform")
st.markdown("An AI-driven platform for retention teams and business leaders to diagnose why customers leave and formulate data-backed retention interventions.")

# ---------------------------------------------------------
# Tabs Setup
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive KPI Dashboard",
    "🧠 Global SHAP Drivers",
    "🔍 Customer Deep-Dive",
    "⚡ Retention What-If Simulator",
    "📁 Batch Scoring & Export"
])

# =========================================================
# TAB 1: Executive KPI Dashboard
# =========================================================
with tab1:
    st.subheader("Executive Portfolio Overview")
    
    # Update predictions based on custom threshold
    custom_pred = (test_df_enriched['Churn Probability'] >= threshold).astype(int)
    total_cust = len(test_df_enriched)
    total_churn_pred = custom_pred.sum()
    churn_rate = (total_churn_pred / total_cust) * 100
    high_risk_count = (test_df_enriched['Churn Probability'] >= 0.65).sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test Accounts Analyzed</div>
            <div class="metric-value">{total_cust:,}</div>
            <div class="metric-delta delta-negative">100% evaluated</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted Churn Rate</div>
            <div class="metric-value">{churn_rate:.1f}%</div>
            <div class="metric-delta delta-positive">Threshold @ {int(threshold*100)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">High-Risk Accounts (&gt;65%)</div>
            <div class="metric-value">{high_risk_count:,}</div>
            <div class="metric-delta delta-positive">Priority retention targets</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Top Churn Indicator</div>
            <div class="metric-value" style="font-size: 1.5rem;">Month-to-Month</div>
            <div class="metric-delta delta-positive">Highest SHAP contribution</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Churn Rate by Contract Duration")
        contract_churn = test_df_enriched.groupby('Contract')['Churn Probability'].mean().reset_index()
        contract_churn['Churn Rate (%)'] = contract_churn['Churn Probability'] * 100
        fig_contract = px.bar(
            contract_churn,
            x='Contract',
            y='Churn Rate (%)',
            color='Contract',
            color_discrete_sequence=['#ef4444', '#3b82f6', '#10b981'],
            text=contract_churn['Churn Rate (%)'].apply(lambda x: f"{x:.1f}%")
        )
        fig_contract.update_layout(template="plotly_dark", height=340, showlegend=False)
        st.plotly_chart(fig_contract, use_container_width=True)

    with chart_col2:
        st.markdown("#### Churn Rate by Internet Service Type")
        net_churn = test_df_enriched.groupby('Internet Service')['Churn Probability'].mean().reset_index()
        net_churn['Churn Rate (%)'] = net_churn['Churn Probability'] * 100
        fig_net = px.bar(
            net_churn,
            x='Internet Service',
            y='Churn Rate (%)',
            color='Internet Service',
            color_discrete_sequence=['#ef4444', '#6366f1', '#10b981'],
            text=net_churn['Churn Rate (%)'].apply(lambda x: f"{x:.1f}%")
        )
        fig_net.update_layout(template="plotly_dark", height=340, showlegend=False)
        st.plotly_chart(fig_net, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("#### Churn Risk vs. Customer Tenure (Months)")
        fig_tenure = px.histogram(
            test_df_enriched,
            x='Tenure Months',
            color='Risk Tier',
            nbins=15,
            color_discrete_map={'High Risk': '#ef4444', 'Moderate Risk': '#f59e0b', 'Low Risk': '#10b981'}
        )
        fig_tenure.update_layout(template="plotly_dark", height=340, barmode='stack')
        st.plotly_chart(fig_tenure, use_container_width=True)

    with chart_col4:
        st.markdown("#### Churn Rate by Payment Method")
        pay_churn = test_df_enriched.groupby('Payment Method')['Churn Probability'].mean().reset_index()
        pay_churn['Churn Rate (%)'] = pay_churn['Churn Probability'] * 100
        pay_churn = pay_churn.sort_values('Churn Rate (%)', ascending=False)
        fig_pay = px.bar(
            pay_churn,
            y='Payment Method',
            x='Churn Rate (%)',
            orientation='h',
            color='Payment Method',
            color_discrete_sequence=['#ef4444', '#f59e0b', '#3b82f6', '#10b981'],
            text=pay_churn['Churn Rate (%)'].apply(lambda x: f"{x:.1f}%")
        )
        fig_pay.update_layout(template="plotly_dark", height=340, showlegend=False)
        st.plotly_chart(fig_pay, use_container_width=True)

# =========================================================
# TAB 2: Global SHAP Drivers
# =========================================================
with tab2:
    st.subheader("Global Explainability: What Drives Churn Across All Accounts?")
    st.markdown("""
    SHAP values break down the model's inner decision tree logic.
    - **Bar Plot**: Ranks features by average overall magnitude of influence.
    - **Beeswarm Plot**: Shows whether high values (red) or low values (blue) increase churn probability.
    """)

    max_display = st.slider("Number of Top Features to Display", min_value=5, max_value=25, value=12)

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.markdown("#### Overall Feature Importance (|SHAP|)")
        fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_values, x_test_transformed, plot_type="bar", max_display=max_display, show=False)
        plt.tight_layout()
        st.pyplot(fig_bar)
        plt.close(fig_bar)

    with shap_col2:
        st.markdown("#### Feature Impact & Direction (Beeswarm)")
        fig_bee, ax_bee = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_values, x_test_transformed, max_display=max_display, show=False)
        plt.tight_layout()
        st.pyplot(fig_bee)
        plt.close(fig_bee)

    st.markdown("---")
    st.subheader("Interactive Feature Dependence Plot")
    st.markdown("Analyze how changes in a specific feature affect churn log-odds, and discover automated interaction with related attributes.")
    
    # Let user pick any feature
    feature_to_plot = st.selectbox(
        "Select Feature for Dependence Analysis",
        options=clean_feature_names,
        index=clean_feature_names.index("Tenure Months") if "Tenure Months" in clean_feature_names else 0
    )

    fig_dep, ax_dep = plt.subplots(figsize=(8, 4.5))
    shap.dependence_plot(feature_to_plot, shap_values, x_test_transformed, ax=ax_dep, show=False)
    plt.tight_layout()
    st.pyplot(fig_dep)
    plt.close(fig_dep)

# =========================================================
# TAB 3: Customer Deep-Dive & Local Explainability
# =========================================================
with tab3:
    st.subheader("Individual Customer Diagnostic & SHAP Waterfall")
    st.markdown("Inspect any customer account to view their predicted churn risk and the exact factors that contributed to that prediction.")

    filter_risk = st.radio("Filter Customer List By Risk:", ["All", "High Risk Only", "Moderate Risk Only", "Low Risk Only"], horizontal=True)
    
    if filter_risk == "High Risk Only":
        filtered_df = test_df_enriched[test_df_enriched['Risk Tier'] == 'High Risk']
    elif filter_risk == "Moderate Risk Only":
        filtered_df = test_df_enriched[test_df_enriched['Risk Tier'] == 'Moderate Risk']
    elif filter_risk == "Low Risk Only":
        filtered_df = test_df_enriched[test_df_enriched['Risk Tier'] == 'Low Risk']
    else:
        filtered_df = test_df_enriched

    selected_idx = st.selectbox(
        "Select Customer Account:",
        options=filtered_df.index,
        format_func=lambda idx: f"ID: {filtered_df.loc[idx].get('CustomerID', idx)} | Churn Risk: {filtered_df.loc[idx]['Churn Probability']*100:.1f}% ({filtered_df.loc[idx]['Risk Tier']})"
    )

    cust_row = test_df_enriched.loc[selected_idx]
    cust_transformed = x_test_transformed.loc[[selected_idx]]
    cust_proba = cust_row['Churn Probability']

    # Display Customer KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Customer Identifier", str(cust_row.get('CustomerID', selected_idx)))
    with c2:
        badge_class = "badge-high" if cust_proba >= 0.65 else ("badge-moderate" if cust_proba >= 0.35 else "badge-low")
        st.markdown(f"**Risk Level**: <span class='{badge_class}'>{cust_row['Risk Tier']}</span>", unsafe_allow_html=True)
    with c3:
        st.metric("Predicted Churn Odds", f"{cust_proba*100:.1f}%")
    with c4:
        st.metric("Actual Status", "Churned" if cust_row['Actual Churn'] == 1 else "Retained")

    st.markdown("---")
    
    cust_col_left, cust_col_right = st.columns([1, 2])
    with cust_col_left:
        st.markdown("#### Account Attributes")
        profile_items = {
            "Tenure": f"{cust_row['Tenure Months']} months",
            "Contract": cust_row['Contract'],
            "Internet Service": cust_row['Internet Service'],
            "Tech Support": cust_row['Tech Support'],
            "Online Security": cust_row['Online Security'],
            "Payment Method": cust_row['Payment Method'],
            "Paperless Billing": cust_row['Paperless Billing'],
            "Dependents": cust_row['Dependents']
        }
        for k, v in profile_items.items():
            st.markdown(f"**{k}**: `{v}`")

    with cust_col_right:
        st.markdown("#### Local SHAP Waterfall Explanation")
        st.caption("Red bars push the customer towards churn (+); Blue bars push towards retention (-).")
        
        # Calculate explanation for this individual customer
        pos_in_test = list(test_df_enriched.index).index(selected_idx)
        single_expl = explainer(cust_transformed)[0]
        
        fig_waterfall, ax_w = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(single_expl, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_waterfall)
        plt.close(fig_waterfall)

# =========================================================
# TAB 4: Retention What-If Simulator
# =========================================================
with tab4:
    st.subheader("⚡ Interactive Retention Intervention Simulator")
    st.markdown("""
    **Client Use Case**: A customer calls support or is flagged at high risk of churning. 
    Use this simulator to test retention offers in real time and see how much the churn probability decreases!
    """)

    sim_col1, sim_col2 = st.columns([1.2, 1.8])

    with sim_col1:
        st.markdown("### 1. Customer Baseline Setup")
        sim_tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=4)
        sim_contract = st.selectbox("Contract Type", options=categorical_options.get('Contract', ['Month-to-month', 'One year', 'Two year']))
        sim_internet = st.selectbox("Internet Service", options=categorical_options.get('Internet Service', ['DSL', 'Fiber optic', 'No']))
        sim_tech = st.selectbox("Tech Support", options=categorical_options.get('Tech Support', ['No', 'Yes', 'No internet service']))
        sim_sec = st.selectbox("Online Security", options=categorical_options.get('Online Security', ['No', 'Yes', 'No internet service']))
        sim_tv = st.selectbox("Streaming TV", options=categorical_options.get('Streaming TV', ['No', 'Yes', 'No internet service']))
        sim_movies = st.selectbox("Streaming Movies", options=categorical_options.get('Streaming Movies', ['No', 'Yes', 'No internet service']))
        sim_pay = st.selectbox("Payment Method", options=categorical_options.get('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']))
        sim_paper = st.selectbox("Paperless Billing", options=categorical_options.get('Paperless Billing', ['Yes', 'No']))
        sim_dep = st.selectbox("Dependents", options=categorical_options.get('Dependents', ['No', 'Yes']))
        sim_city = st.selectbox("City", options=categorical_options.get('City', ['Los Angeles'])[:10])

        # Assemble dataframe
        sim_df = pd.DataFrame([{
            'City': sim_city,
            'Dependents': sim_dep,
            'Tenure Months': sim_tenure,
            'Internet Service': sim_internet,
            'Online Security': sim_sec,
            'Tech Support': sim_tech,
            'Streaming TV': sim_tv,
            'Streaming Movies': sim_movies,
            'Contract': sim_contract,
            'Paperless Billing': sim_paper,
            'Payment Method': sim_pay
        }])

    with sim_col2:
        st.markdown("### 2. Live Churn Risk Assessment")
        
        # Preprocess and score
        sim_transformed = pd.DataFrame(
            preprocessor.transform(sim_df),
            columns=clean_feature_names
        )
        base_proba = model.predict_proba(sim_transformed)[0, 1]

        # Metric display
        gauge_col, status_col = st.columns(2)
        with gauge_col:
            st.metric("Current Churn Risk", f"{base_proba*100:.1f}%")
        with status_col:
            if base_proba >= 0.65:
                st.error("🚨 HIGH CHURN RISK: Immediate action required")
            elif base_proba >= 0.35:
                st.warning("⚠️ MODERATE RISK: Account is at risk of leaving")
            else:
                st.success("✅ LOW RISK: Account is currently healthy")

        st.markdown("---")
        st.markdown("### 3. Test Retention Offer Interventions")
        
        c_offer1, c_offer2, c_offer3 = st.columns(3)
        with c_offer1:
            offer_contract = st.checkbox("Offer 1-Year Contract Discount", value=False)
        with c_offer2:
            offer_support = st.checkbox("Include Free Tech Support & Security", value=False)
        with c_offer3:
            offer_autopay = st.checkbox("Switch to Auto Credit Card / Bank", value=False)

        # Build modified dataframe
        mod_df = sim_df.copy()
        if offer_contract and mod_df.at[0, 'Contract'] == 'Month-to-month':
            mod_df.at[0, 'Contract'] = 'One year'
        if offer_support:
            mod_df.at[0, 'Tech Support'] = 'Yes'
            mod_df.at[0, 'Online Security'] = 'Yes'
        if offer_autopay:
            mod_df.at[0, 'Payment Method'] = 'Credit card (automatic)'

        mod_transformed = pd.DataFrame(
            preprocessor.transform(mod_df),
            columns=clean_feature_names
        )
        new_proba = model.predict_proba(mod_transformed)[0, 1]
        delta_risk = (new_proba - base_proba) * 100

        res1, res2 = st.columns(2)
        with res1:
            st.metric("New Churn Risk with Offers", f"{new_proba*100:.1f}%", f"{delta_risk:.1f}%", delta_color="inverse")
        with res2:
            saved_pts = -delta_risk
            if saved_pts > 0:
                st.success(f"🎉 **Save Impact**: Churn probability dropped by **{saved_pts:.1f} percentage points**!")
            else:
                st.info("Check retention offer boxes above to simulate intervention impact.")

        # Real-time Waterfall explanation for simulated account
        st.markdown("#### Live Factor Contribution for this Account")
        sim_expl = explainer(sim_transformed)[0]
        fig_sim_w, ax_sw = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(sim_expl, max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig_sim_w)
        plt.close(fig_sim_w)

# =========================================================
# TAB 5: Batch Scoring & Export
# =========================================================
with tab5:
    st.subheader("📁 Batch Customer Churn Scoring")
    st.markdown("""
    Upload a list of customer accounts (.csv or .xlsx) to score churn risk in bulk, assign priority retention tiers, and export enriched reports for customer success teams.
    """)

    uploaded_file = st.file_uploader("Upload Customer File (CSV or Excel)", type=["csv", "xlsx"])

    batch_df = None
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".xlsx"):
                batch_df = pd.read_excel(uploaded_file)
            else:
                batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df):,} accounts from uploaded file!")
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("No file uploaded. You can load a sample test batch from the IBM Telco dataset to preview:")
        if st.button("Load Sample 100 Customer Batch"):
            batch_df = x_test.head(100).copy()

    if batch_df is not None:
        # Check required columns
        required_cols = list(x_test.columns)
        missing_cols = [c for c in required_cols if c not in batch_df.columns]
        
        if missing_cols:
            st.warning(f"Note: The file is missing some expected columns: {missing_cols}. Missing values will be imputed with defaults.")
            for col in missing_cols:
                if col in categorical_options:
                    batch_df[col] = categorical_options[col][0]
                else:
                    batch_df[col] = 0

        # Transform and predict
        batch_transformed = pd.DataFrame(
            preprocessor.transform(batch_df[required_cols]),
            columns=clean_feature_names,
            index=batch_df.index
        )
        batch_probas = model.predict_proba(batch_transformed)[:, 1]
        batch_preds = (batch_probas >= threshold).astype(int)

        scored_df = batch_df.copy()
        scored_df['Churn Probability'] = np.round(batch_probas, 4)
        scored_df['Predicted Churn'] = batch_preds
        scored_df['Risk Tier'] = pd.cut(
            scored_df['Churn Probability'],
            bins=[-0.01, 0.35, 0.65, 1.01],
            labels=['Low Risk', 'Moderate Risk', 'High Risk']
        )

        st.markdown("#### Scored Accounts Preview")
        st.dataframe(
            scored_df[['Tenure Months', 'Contract', 'Internet Service', 'Payment Method', 'Churn Probability', 'Risk Tier']].head(20),
            use_container_width=True
        )

        # Export options
        csv_data = scored_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Scored Accounts CSV",
            data=csv_data,
            file_name="telco_scored_churn_accounts.csv",
            mime="text/csv"
        )
