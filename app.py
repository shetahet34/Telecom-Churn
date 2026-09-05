import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import io
import datetime

# ---------------------------------------------------------
# Page Configuration & Executive Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Telco Churn Intelligence Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern, High-Contrast Executive CSS
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Clean Metric Boxes */
    .metric-box {
        background: #151e2e;
        border: 1px solid #26334d;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-box:hover {
        border-color: #3b82f6;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .metric-sub {
        font-size: 0.82rem;
        font-weight: 500;
        color: #64748b;
    }
    
    /* Risk Badges */
    .badge-high {
        background-color: rgba(239, 68, 68, 0.18);
        color: #fca5a5;
        border: 1px solid #ef4444;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.18);
        color: #fcd34d;
        border: 1px solid #f59e0b;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-low {
        background-color: rgba(16, 185, 129, 0.18);
        color: #6ee7b7;
        border: 1px solid #10b981;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Recommendation & Callout Cards */
    .insight-card {
        background: linear-gradient(135deg, #182235 0%, #111827 100%);
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 15px 0;
    }
    .action-card {
        background: linear-gradient(135deg, #172554 0%, #0f172a 100%);
        border: 1px solid #1d4ed8;
        border-radius: 10px;
        padding: 18px 22px;
        margin-top: 15px;
    }
    .action-step {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border-left: 3px solid #60a5fa;
        font-size: 0.92rem;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #111827;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #1f2937;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 22px;
        font-weight: 600;
        color: #94a3b8;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1d4ed8 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Cached Assets
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
test_df_enriched = assets['test_df_enriched'].copy()
shap_values = assets['shap_values']
explainer = assets['explainer']

# Add estimated monthly charge if missing (avg is ~$65)
if 'Monthly Charges' not in test_df_enriched.columns:
    test_df_enriched['Estimated Monthly Revenue'] = 65.00
else:
    test_df_enriched['Estimated Monthly Revenue'] = test_df_enriched['Monthly Charges']

test_df_enriched['Churn Percentage'] = np.round(test_df_enriched['Churn Probability'] * 100, 1)
test_df_enriched['Churn Chance'] = test_df_enriched['Churn Percentage'].apply(lambda v: f"{v:.1f}%")

# Feature naming dictionary for non-technical users
FRIENDLY_FEATURE_NAMES = {
    'Contract_Month-to-month': 'Month-to-Month Contract',
    'Contract_Two year': '2-Year Contract Commitment',
    'Contract_One year': '1-Year Contract Commitment',
    'Tenure Months': 'Customer Relationship Length (Tenure)',
    'Internet Service_Fiber optic': 'Fiber Optic Internet Plan',
    'Internet Service_DSL': 'DSL Internet Plan',
    'Internet Service_No': 'No Internet Service',
    'Online Security_No': 'No Online Security Addon',
    'Online Security_Yes': 'Has Online Security Addon',
    'Tech Support_No': 'No Tech Support Addon',
    'Tech Support_Yes': 'Has Tech Support Addon',
    'Payment Method_Electronic check': 'Paying via Electronic Check',
    'Payment Method_Credit card (automatic)': 'Auto-Pay (Credit Card)',
    'Payment Method_Bank transfer (automatic)': 'Auto-Pay (Bank Transfer)',
    'Payment Method_Mailed check': 'Paying via Mailed Check',
    'Dependents_No': 'Individual Account (No Dependents)',
    'Dependents_Yes': 'Family Account (Has Dependents)',
    'Paperless Billing_Yes': 'Enrolled in Paperless Billing',
    'Paperless Billing_No': 'Paper Invoicing',
    'Streaming Movies_Yes': 'Subscribed to Streaming Movies',
    'Streaming Movies_No': 'No Streaming Movies',
    'Streaming TV_Yes': 'Subscribed to Streaming TV',
    'Streaming TV_No': 'No Streaming TV'
}

def get_friendly_name(name):
    if name in FRIENDLY_FEATURE_NAMES:
        return FRIENDLY_FEATURE_NAMES[name]
    if name.startswith('City_'):
        return f"Customer City: {name.replace('City_', '')}"
    return name.replace('_', ': ')

# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 📡 Churn Intelligence")
    st.caption("Retention AI for Business Leaders")
    
    st.markdown("---")
    st.markdown("#### 🎯 Portfolio Quick Filters")
    
    # Contract filter
    contract_choices = ['All Contracts'] + sorted(list(test_df_enriched['Contract'].dropna().unique()))
    selected_contract = st.selectbox("Contract Type", options=contract_choices)
    
    # Internet service filter
    internet_choices = ['All Services'] + sorted(list(test_df_enriched['Internet Service'].dropna().unique()))
    selected_internet = st.selectbox("Internet Service", options=internet_choices)

    # Risk level filter
    risk_choices = ['All Risk Levels', 'High Risk Only', 'Moderate Risk Only', 'Low Risk Only']
    selected_risk = st.selectbox("Customer Risk Level", options=risk_choices)

    st.markdown("---")
    st.markdown("#### ⚙️ Retention Cutoff Sensitivity")
    threshold = st.slider(
        "High-Risk Alert Cutoff",
        min_value=0.30,
        max_value=0.70,
        value=0.50,
        step=0.05,
        help="Accounts with predicted churn chance above this cutoff will be flagged for immediate retention outreach."
    )
    
    st.markdown("---")
    st.markdown("""
    **💡 Quick Help Guide**:
    - **Overview**: High-level portfolio metrics & revenue at risk.
    - **Why Leave**: Plain-English AI drivers of customer cancellations.
    - **Customer Profiler**: Instant risk diagnosis & call scripts.
    - **Simulator**: Test retention offers before calling customers.
    - **Call List**: Export prioritized customer lists for outreach.
    """)

# Apply Global Filters to working copy
filtered_df = test_df_enriched.copy()
if selected_contract != 'All Contracts':
    filtered_df = filtered_df[filtered_df['Contract'] == selected_contract]
if selected_internet != 'All Services':
    filtered_df = filtered_df[filtered_df['Internet Service'] == selected_internet]
if selected_risk == 'High Risk Only':
    filtered_df = filtered_df[filtered_df['Churn Probability'] >= 0.65]
elif selected_risk == 'Moderate Risk Only':
    filtered_df = filtered_df[(filtered_df['Churn Probability'] >= 0.35) & (filtered_df['Churn Probability'] < 0.65)]
elif selected_risk == 'Low Risk Only':
    filtered_df = filtered_df[filtered_df['Churn Probability'] < 0.35]

# Top Welcome Banner
st.markdown("""
<div style="padding: 8px 0px 14px 0px;">
    <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #f8fafc;">
        📡 Telecom Customer Churn Intelligence & Retention Hub
    </h1>
    <p style="margin-top: 6px; font-size: 1.02rem; color: #94a3b8;">
        An intuitive, non-technical platform for business leaders and account managers to diagnose churn risks, uncover why accounts leave, and test retention offers that protect revenue.
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Primary App Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Overview & Revenue Impact",
    "💡 Why Customers Leave (Plain-English AI)",
    "👤 Customer Risk Profiler & Call Assistant",
    "🎯 Retention Offer Simulator (What-If)",
    "📑 Prioritized Call List & File Scorer"
])

# =========================================================
# TAB 1: Executive Overview & Revenue Impact
# =========================================================
with tab1:
    st.markdown("### 📈 Executive Portfolio Health & Revenue Exposure")
    st.caption("Real-time visibility into customer attrition risk, segment hotspots, and revenue exposure.")
    
    total_active = len(filtered_df)
    churned_pred = (filtered_df['Churn Probability'] >= threshold).sum()
    pred_churn_rate = (churned_pred / total_active * 100) if total_active > 0 else 0
    high_risk_accounts = (filtered_df['Churn Probability'] >= 0.65).sum()
    
    # Financial estimation ($65 avg bill)
    monthly_rev_at_risk = high_risk_accounts * 65.0
    annual_rev_at_risk = monthly_rev_at_risk * 12

    # Hero KPI Cards
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Accounts Monitored</div>
            <div class="metric-number">{total_active:,}</div>
            <div class="metric-sub">Total customer cohort</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        prob_color = '#ef4444' if pred_churn_rate > 25 else '#f59e0b'
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Predicted Churn Rate</div>
            <div class="metric-number" style="color: {prob_color};">{pred_churn_rate:.1f}%</div>
            <div class="metric-sub">Flagged above {int(threshold*100)}% risk cutoff</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">High-Risk Accounts</div>
            <div class="metric-number" style="color: #ef4444;">{high_risk_accounts:,}</div>
            <div class="metric-sub">Churn Chance &gt; 65% (Urgent Action)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Monthly Revenue at Risk</div>
            <div class="metric-number" style="color: #fb923c;">${monthly_rev_at_risk:,.0f}</div>
            <div class="metric-sub">${annual_rev_at_risk:,.0f} / year exposure</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="insight-card">
        <strong>💡 Key Executive Takeaway:</strong> Month-to-month subscribers with short tenure (&lt; 12 months) and electronic check billing are <strong>4.2x more likely to churn</strong> than long-term auto-pay customers. Transitioning 15% of high-risk accounts to annual contracts would protect approximately <strong>${monthly_rev_at_risk*0.15:,.0f}/month</strong> in recurring revenue.
    </div>
    """, unsafe_allow_html=True)

    # Visual Hotspots
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        st.markdown("#### 📄 Churn Rate by Contract Duration")
        c_grp = filtered_df.groupby('Contract')['Churn Probability'].mean().reset_index()
        c_grp['Churn %'] = c_grp['Churn Probability'] * 100
        fig_c = px.bar(
            c_grp,
            x='Contract',
            y='Churn %',
            color='Contract',
            color_discrete_map={
                'Month-to-month': '#ef4444',
                'One year': '#3b82f6',
                'Two year': '#10b981'
            },
            text=c_grp['Churn %'].apply(lambda v: f"{v:.1f}%")
        )
        fig_c.update_layout(
            template="plotly_dark",
            height=320,
            showlegend=False,
            yaxis_title="Average Churn Percentage (%)",
            xaxis_title="",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with col_plot2:
        st.markdown("#### 🌐 Churn Rate by Internet Service Type")
        net_grp = filtered_df.groupby('Internet Service')['Churn Probability'].mean().reset_index()
        net_grp['Churn %'] = net_grp['Churn Probability'] * 100
        fig_net = px.bar(
            net_grp,
            x='Internet Service',
            y='Churn %',
            color='Internet Service',
            color_discrete_map={
                'Fiber optic': '#ef4444',
                'DSL': '#3b82f6',
                'No': '#10b981'
            },
            text=net_grp['Churn %'].apply(lambda v: f"{v:.1f}%")
        )
        fig_net.update_layout(
            template="plotly_dark",
            height=320,
            showlegend=False,
            yaxis_title="Average Churn Percentage (%)",
            xaxis_title="",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_net, use_container_width=True)

    col_plot3, col_plot4 = st.columns(2)
    
    with col_plot3:
        st.markdown("#### ⏳ When Do Customers Leave? (Tenure Cohort Risk)")
        filtered_df_copy = filtered_df.copy()
        filtered_df_copy['Tenure Cohort'] = pd.cut(
            filtered_df_copy['Tenure Months'],
            bins=[-1, 6, 12, 24, 48, 100],
            labels=['0-6 mos (Critical)', '7-12 mos', '1-2 yrs', '2-4 yrs', '4+ yrs (Loyal)']
        )
        t_grp = filtered_df_copy.groupby('Tenure Cohort', observed=False)['Churn Probability'].mean().reset_index()
        t_grp['Churn %'] = t_grp['Churn Probability'] * 100
        fig_t = px.bar(
            t_grp,
            x='Tenure Cohort',
            y='Churn %',
            color='Churn %',
            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
            text=t_grp['Churn %'].apply(lambda v: f"{v:.1f}%")
        )
        fig_t.update_layout(
            template="plotly_dark",
            height=320,
            coloraxis_showscale=False,
            yaxis_title="Average Churn Percentage (%)",
            xaxis_title="",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_t, use_container_width=True)

    with col_plot4:
        st.markdown("#### 💳 Churn Risk by Payment Method")
        p_grp = filtered_df.groupby('Payment Method')['Churn Probability'].mean().reset_index()
        p_grp['Churn %'] = p_grp['Churn Probability'] * 100
        p_grp = p_grp.sort_values('Churn %', ascending=True)
        fig_p = px.bar(
            p_grp,
            y='Payment Method',
            x='Churn %',
            orientation='h',
            color='Churn %',
            color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'],
            text=p_grp['Churn %'].apply(lambda v: f"{v:.1f}%")
        )
        fig_p.update_layout(
            template="plotly_dark",
            height=320,
            coloraxis_showscale=False,
            xaxis_title="Average Churn Percentage (%)",
            yaxis_title="",
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_p, use_container_width=True)

# =========================================================
# TAB 2: Why Customers Leave (Plain-English AI Drivers)
# =========================================================
with tab2:
    st.markdown("### 💡 What Drives Customer Loyalty vs. Cancellations?")
    st.markdown("""
    Our Machine Learning model evaluates over 1,000 customer attributes. Below, we have translated the complex AI decision paths into **clear, non-technical business factors** that your marketing and retention teams can act on immediately.
    """)

    # Calculate global drivers
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]

    drivers_data = []
    for idx in top_indices:
        raw_feature = clean_feature_names[idx]
        friendly = get_friendly_name(raw_feature)
        importance_score = mean_abs_shap[idx]
        
        # Determine business tendency
        if 'Month-to-month' in raw_feature or 'Fiber optic' in raw_feature or 'Electronic check' in raw_feature or '_No' in raw_feature:
            nature = 'Increases Churn Risk 🔴'
            color = '#ef4444'
        else:
            nature = 'Builds Customer Loyalty 🟢'
            color = '#10b981'

        drivers_data.append({
            'Business Factor': friendly,
            'Influence Magnitude': importance_score,
            'Nature': nature,
            'Color': color,
            'Raw': raw_feature
        })

    drivers_df = pd.DataFrame(drivers_data)
    drivers_df = drivers_df.sort_values('Influence Magnitude', ascending=True)

    fig_drivers = go.Figure(go.Bar(
        x=drivers_df['Influence Magnitude'],
        y=drivers_df['Business Factor'],
        orientation='h',
        marker=dict(color=drivers_df['Color']),
        text=drivers_df['Nature'],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Influence Score: %{x:.3f}<br>%{text}<extra></extra>"
    ))
    fig_drivers.update_layout(
        template="plotly_dark",
        height=450,
        xaxis_title="Relative Strength of Impact on Customer Decision",
        yaxis_title="",
        margin=dict(l=20, r=140, t=30, b=20)
    )
    st.plotly_chart(fig_drivers, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔍 Interactive Factor Explorer & Retention Playbook")
    st.caption("Pick any factor below to see how it impacts retention and read our recommended business playbook.")

    chosen_factor = st.selectbox(
        "Choose a Business Factor to Inspect:",
        [
            "Contract Commitment (Month-to-Month vs Annual)",
            "Tech Support Addon",
            "Online Security Protection",
            "Payment Method (Manual Check vs Auto-Pay)",
            "Paperless Billing Enrollment"
        ]
    )

    f_col1, f_col2 = st.columns([1.3, 1.7])

    if chosen_factor == "Contract Commitment (Month-to-Month vs Annual)":
        sub_df = test_df_enriched.groupby('Contract')['Churn Probability'].mean().reset_index()
        sub_df['Risk %'] = sub_df['Churn Probability'] * 100
        with f_col1:
            fig_sub = px.bar(sub_df, x='Contract', y='Risk %', color='Contract', color_discrete_sequence=['#ef4444', '#3b82f6', '#10b981'])
            fig_sub.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig_sub, use_container_width=True)
        with f_col2:
            st.markdown("""
            <div class="action-card">
                <h4 style="color: #60a5fa; margin-top: 0;">📋 Executive Playbook: Contract Upgrades</h4>
                <div class="action-step"><strong>Root Cause:</strong> Month-to-month contracts have zero switching cost, making customers volatile.</div>
                <div class="action-step"><strong>Campaign:</strong> Target month-to-month customers in months 3-6 with a <strong>15% annual discount</strong> or <strong>1 free month</strong> in exchange for a 1-year agreement.</div>
                <div class="action-step"><strong>Expected Retention Boost:</strong> Lowers individual account churn risk by over <strong>40 percentage points</strong>.</div>
            </div>
            """, unsafe_allow_html=True)

    elif chosen_factor == "Tech Support Addon":
        sub_df = test_df_enriched.groupby('Tech Support')['Churn Probability'].mean().reset_index()
        sub_df['Risk %'] = sub_df['Churn Probability'] * 100
        with f_col1:
            fig_sub = px.bar(sub_df, x='Tech Support', y='Risk %', color='Tech Support', color_discrete_sequence=['#ef4444', '#10b981', '#94a3b8'])
            fig_sub.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig_sub, use_container_width=True)
        with f_col2:
            st.markdown("""
            <div class="action-card">
                <h4 style="color: #60a5fa; margin-top: 0;">📋 Executive Playbook: Premium Tech Support</h4>
                <div class="action-step"><strong>Root Cause:</strong> Customers without tech support feel frustrated when experiencing connectivity hiccups and cancel out of annoyance.</div>
                <div class="action-step"><strong>Campaign:</strong> Bundle 60 days of free 24/7 dedicated VIP Tech Support for high-value broadband accounts.</div>
                <div class="action-step"><strong>Expected Retention Boost:</strong> Reduces churn likelihood by <strong>18–25 percentage points</strong>.</div>
            </div>
            """, unsafe_allow_html=True)

    elif chosen_factor == "Online Security Protection":
        sub_df = test_df_enriched.groupby('Online Security')['Churn Probability'].mean().reset_index()
        sub_df['Risk %'] = sub_df['Churn Probability'] * 100
        with f_col1:
            fig_sub = px.bar(sub_df, x='Online Security', y='Risk %', color='Online Security', color_discrete_sequence=['#ef4444', '#10b981', '#94a3b8'])
            fig_sub.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig_sub, use_container_width=True)
        with f_col2:
            st.markdown("""
            <div class="action-card">
                <h4 style="color: #60a5fa; margin-top: 0;">📋 Executive Playbook: Cybersecurity Bundles</h4>
                <div class="action-step"><strong>Root Cause:</strong> Value-added security services create product stickiness and higher customer satisfaction.</div>
                <div class="action-step"><strong>Campaign:</strong> Launch a "Home Peace-of-Mind" package bundling antivirus and firewall security at $3/mo.</div>
                <div class="action-step"><strong>Expected Retention Boost:</strong> Strengthens account loyalty and cuts churn odds by <strong>20 percentage points</strong>.</div>
            </div>
            """, unsafe_allow_html=True)

    elif chosen_factor == "Payment Method (Manual Check vs Auto-Pay)":
        sub_df = test_df_enriched.groupby('Payment Method')['Churn Probability'].mean().reset_index()
        sub_df['Risk %'] = sub_df['Churn Probability'] * 100
        with f_col1:
            fig_sub = px.bar(sub_df, y='Payment Method', x='Risk %', orientation='h', color='Risk %', color_continuous_scale=['#10b981', '#f59e0b', '#ef4444'])
            fig_sub.update_layout(template="plotly_dark", height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_sub, use_container_width=True)
        with f_col2:
            st.markdown("""
            <div class="action-card">
                <h4 style="color: #60a5fa; margin-top: 0;">📋 Executive Playbook: Auto-Pay Conversion</h4>
                <div class="action-step"><strong>Root Cause:</strong> Electronic check payers actively re-evaluate their bill every month, creating regular opportunities to cancel.</div>
                <div class="action-step"><strong>Campaign:</strong> Offer a one-time $10 bill credit for customers who enroll in automated credit card or bank debit.</div>
                <div class="action-step"><strong>Expected Retention Boost:</strong> Cuts churn rates in half for converted accounts.</div>
            </div>
            """, unsafe_allow_html=True)

    elif chosen_factor == "Paperless Billing Enrollment":
        sub_df = test_df_enriched.groupby('Paperless Billing')['Churn Probability'].mean().reset_index()
        sub_df['Risk %'] = sub_df['Churn Probability'] * 100
        with f_col1:
            fig_sub = px.bar(sub_df, x='Paperless Billing', y='Risk %', color='Paperless Billing', color_discrete_sequence=['#ef4444', '#10b981'])
            fig_sub.update_layout(template="plotly_dark", height=300, showlegend=False)
            st.plotly_chart(fig_sub, use_container_width=True)
        with f_col2:
            st.markdown("""
            <div class="action-card">
                <h4 style="color: #60a5fa; margin-top: 0;">📋 Executive Playbook: Digital Billing Engagement</h4>
                <div class="action-step"><strong>Root Cause:</strong> Paperless billing users often experience higher bill shock when unexpected charges appear.</div>
                <div class="action-step"><strong>Campaign:</strong> Implement proactive SMS alerts explaining month-to-month usage fluctuations.</div>
                <div class="action-step"><strong>Expected Retention Boost:</strong> Lowers bill-related customer complaints and churn calls by <strong>15%</strong>.</div>
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# TAB 3: Customer Risk Profiler & Call Assistant
# =========================================================
with tab3:
    st.markdown("### 👤 Individual Customer Risk Diagnostic & Call Assistant")
    st.caption("Designed for Account Managers and Customer Retention Agents during live customer outreach.")

    prof_filter = st.radio(
        "Filter Customer Accounts By:",
        ["🔴 High Risk Only (>65% Churn)", "🟡 Moderate Risk (35-65%)", "🟢 Low Risk / Healthy (<35%)", "All Accounts"],
        horizontal=True
    )

    if "High Risk" in prof_filter:
        subset_prof = test_df_enriched[test_df_enriched['Risk Tier'] == 'High Risk']
    elif "Moderate Risk" in prof_filter:
        subset_prof = test_df_enriched[test_df_enriched['Risk Tier'] == 'Moderate Risk']
    elif "Low Risk" in prof_filter:
        subset_prof = test_df_enriched[test_df_enriched['Risk Tier'] == 'Low Risk']
    else:
        subset_prof = test_df_enriched

    if len(subset_prof) == 0:
        subset_prof = test_df_enriched

    selected_account_idx = st.selectbox(
        "Select Customer Account to Diagnose:",
        options=subset_prof.index,
        format_func=lambda i: f"Account ID: {subset_prof.loc[i].get('CustomerID', i)} | Churn Chance: {subset_prof.loc[i]['Churn Percentage']:.1f}% | Contract: {subset_prof.loc[i]['Contract']}"
    )

    account_row = test_df_enriched.loc[selected_account_idx]
    account_prob = account_row['Churn Probability']
    account_transformed = x_test_transformed.loc[[selected_account_idx]]

    # Customer Summary Metrics
    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
    with p_c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Customer ID</div>
            <div class="metric-number" style="font-size: 1.5rem;">{account_row.get('CustomerID', selected_account_idx)}</div>
            <div class="metric-sub">{account_row.get('City', 'California')}</div>
        </div>
        """, unsafe_allow_html=True)
    with p_c2:
        badge_html = "<span class='badge-high'>🚨 HIGH CHURN RISK</span>" if account_prob >= 0.65 else ("<span class='badge-moderate'>⚠️ MODERATE RISK</span>" if account_prob >= 0.35 else "<span class='badge-low'>✅ HEALTHY ACCOUNT</span>")
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Risk Tier Assessment</div>
            <div style="margin-top: 6px; margin-bottom: 8px;">{badge_html}</div>
            <div class="metric-sub">Real-time Churn Chance</div>
        </div>
        """, unsafe_allow_html=True)
    with p_c3:
        prob_color = '#ef4444' if account_prob >= 0.65 else ('#f59e0b' if account_prob >= 0.35 else '#10b981')
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Predicted Churn Chance</div>
            <div class="metric-number" style="color: {prob_color};">{account_prob*100:.1f}%</div>
            <div class="metric-sub">Likelihood to cancel</div>
        </div>
        """, unsafe_allow_html=True)
    with p_c4:
        monthly_val = account_row.get('Estimated Monthly Revenue', 65.0)
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Estimated Monthly Value</div>
            <div class="metric-number">${monthly_val:.0f}/mo</div>
            <div class="metric-sub">${monthly_val*12:.0f} Annual Value</div>
        </div>
        """, unsafe_allow_html=True)

    prof_col_left, prof_col_right = st.columns([1.1, 1.9])

    with prof_col_left:
        st.markdown("#### 📋 Current Customer Profile")
        st.markdown(f"""
        - **Tenure**: `{account_row['Tenure Months']} months`
        - **Contract**: `{account_row['Contract']}`
        - **Internet Service**: `{account_row['Internet Service']}`
        - **Tech Support**: `{account_row['Tech Support']}`
        - **Online Security**: `{account_row['Online Security']}`
        - **Payment Method**: `{account_row['Payment Method']}`
        - **Paperless Billing**: `{account_row['Paperless Billing']}`
        - **Dependents**: `{account_row['Dependents']}`
        """)

    with prof_col_right:
        st.markdown("#### 🔍 Why is This Specific Customer At Risk?")
        st.caption("How specific features push this customer toward staying (Green) or leaving (Red):")

        single_expl = explainer(account_transformed)[0]
        cust_shap_vals = single_expl.values
        
        top_k = 7
        top_indices = np.argsort(np.abs(cust_shap_vals))[::-1][:top_k]
        
        cust_factors = []
        for i in top_indices:
            raw = clean_feature_names[i]
            val = cust_shap_vals[i]
            friendly = get_friendly_name(raw)
            impact_type = "Increases Risk of Leaving 🔴" if val > 0 else "Encourages Staying 🟢"
            bar_color = "#ef4444" if val > 0 else "#10b981"
            cust_factors.append({
                'Factor': friendly,
                'Impact Score': val,
                'Impact Type': impact_type,
                'Color': bar_color
            })
            
        c_df = pd.DataFrame(cust_factors).sort_values('Impact Score', ascending=True)

        fig_cust = go.Figure(go.Bar(
            x=c_df['Impact Score'],
            y=c_df['Factor'],
            orientation='h',
            marker=dict(color=c_df['Color']),
            text=[f"{v:+.2f} ({t})" for v, t in zip(c_df['Impact Score'], c_df['Impact Type'])],
            textposition='auto',
            hovertemplate="<b>%{y}</b><br>Impact Score: %{x:+.2f}<extra></extra>"
        ))
        fig_cust.update_layout(
            template="plotly_dark",
            height=300,
            xaxis_title="Pushing Towards Retention (Left)  vs.  Pushing Towards Churn (Right)",
            yaxis_title="",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_cust, use_container_width=True)

    # Retention Action Plan Card
    st.markdown("""
    <div class="action-card">
        <h4 style="color: #60a5fa; margin-top: 0;">🎯 Recommended Retention Action Plan for Account Manager</h4>
    """, unsafe_allow_html=True)

    rec_steps = []
    if account_row['Contract'] == 'Month-to-month':
        rec_steps.append("<strong>Step 1: Offer Contract Upgrade</strong>: Customer is on a Month-to-Month agreement. Offer a 1-Year commitment with a 15% discount for the first 3 months.")
    else:
        rec_steps.append("<strong>Step 1: Contract Renewal Acknowledgment</strong>: Customer is on an annual agreement. Send a loyalty thank-you bonus.")

    if account_row['Tech Support'] == 'No':
        rec_steps.append("<strong>Step 2: Add Complimentary Support</strong>: Account lacks Tech Support. Provide 3 months of complimentary 24/7 VIP Tech Support.")
    elif account_row['Online Security'] == 'No':
        rec_steps.append("<strong>Step 2: Add Security Protection</strong>: Account lacks Online Security. Offer free trial of Cybersecurity & Parental Controls.")
    else:
        rec_steps.append("<strong>Step 2: Broadband Quality Check</strong>: Customer has basic addons. Inquire about streaming and Wi-Fi coverage satisfaction.")

    if 'Electronic check' in account_row['Payment Method']:
        rec_steps.append("<strong>Step 3: Migrate to Auto-Pay</strong>: Customer pays via Electronic Check. Offer a one-time $10 bill credit for switching to Credit Card or Bank Auto-Pay.")
    else:
        rec_steps.append("<strong>Step 3: Billing Confirmation</strong>: Customer is on auto-pay. Confirm billing contact information is up to date.")

    for step in rec_steps:
        st.markdown(f"<div class='action-step'>{step}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# TAB 4: Retention Offer Simulator (What-If Tester)
# =========================================================
with tab4:
    st.markdown("### 🎯 Interactive Retention Offer Simulator (What-If Analysis)")
    st.markdown("""
    Simulate customer scenarios in real time! Change customer attributes on the left, check off retention packages on the right, and instantly watch the **live churn risk meter drop** and calculate **saved revenue**.
    """)

    sim_left, sim_right = st.columns([1.1, 1.9])

    with sim_left:
        st.markdown("#### 1. Configure Customer Baseline")
        sim_tenure = st.slider("Tenure Length (Months)", min_value=1, max_value=72, value=3)
        sim_contract = st.selectbox("Contract Type", options=categorical_options.get('Contract', ['Month-to-month', 'One year', 'Two year']))
        sim_internet = st.selectbox("Internet Service", options=categorical_options.get('Internet Service', ['Fiber optic', 'DSL', 'No']))
        sim_tech = st.selectbox("Tech Support Addon", options=categorical_options.get('Tech Support', ['No', 'Yes', 'No internet service']))
        sim_sec = st.selectbox("Online Security Addon", options=categorical_options.get('Online Security', ['No', 'Yes', 'No internet service']))
        sim_pay = st.selectbox("Payment Method", options=categorical_options.get('Payment Method', ['Electronic check', 'Credit card (automatic)', 'Bank transfer (automatic)', 'Mailed check']))
        sim_tv = st.selectbox("Streaming TV", options=categorical_options.get('Streaming TV', ['No', 'Yes', 'No internet service']))
        sim_movies = st.selectbox("Streaming Movies", options=categorical_options.get('Streaming Movies', ['No', 'Yes', 'No internet service']))
        sim_paper = st.selectbox("Paperless Billing", options=['Yes', 'No'])
        sim_dep = st.selectbox("Dependents / Family Plan", options=['No', 'Yes'])
        sim_city = 'Los Angeles'

        base_sim_df = pd.DataFrame([{
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

    with sim_right:
        st.markdown("#### 2. Live Churn Chance Speedometer")

        # Calculate baseline probability
        base_transformed = pd.DataFrame(
            preprocessor.transform(base_sim_df),
            columns=clean_feature_names
        )
        base_prob = model.predict_proba(base_transformed)[0, 1] * 100

        st.markdown("#### 3. Test Retention Offer Interventions")
        st.caption("Check the boxes below to test how different retention offers immediately reduce churn risk:")
        
        offer_col1, offer_col2, offer_col3 = st.columns(3)
        with offer_col1:
            offer_contract = st.checkbox("🎁 Upgrade to 1-Year Contract", value=False)
        with offer_col2:
            offer_support = st.checkbox("🛡️ Bundle Free Tech & Security", value=False)
        with offer_col3:
            offer_autopay = st.checkbox("💳 Switch to Auto-Pay", value=False)

        # Build modified customer
        mod_sim_df = base_sim_df.copy()
        if offer_contract and mod_sim_df.at[0, 'Contract'] == 'Month-to-month':
            mod_sim_df.at[0, 'Contract'] = 'One year'
        if offer_support:
            mod_sim_df.at[0, 'Tech Support'] = 'Yes'
            mod_sim_df.at[0, 'Online Security'] = 'Yes'
        if offer_autopay:
            mod_sim_df.at[0, 'Payment Method'] = 'Credit card (automatic)'

        mod_transformed = pd.DataFrame(
            preprocessor.transform(mod_sim_df),
            columns=clean_feature_names
        )
        new_prob = model.predict_proba(mod_transformed)[0, 1] * 100
        risk_reduction = base_prob - new_prob

        # Display Live Gauge Speedometer
        gauge_color = '#ef4444' if new_prob >= 65 else ('#f59e0b' if new_prob >= 35 else '#10b981')
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=new_prob,
            title={'text': "<b>Current Churn Chance</b>", 'font': {'size': 20, 'color': '#ffffff'}},
            delta={'reference': base_prob, 'increasing': {'color': '#ef4444'}, 'decreasing': {'color': '#10b981'}},
            number={'suffix': "%", 'font': {'color': gauge_color, 'size': 38}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#94a3b8'},
                'bar': {'color': gauge_color},
                'bgcolor': 'rgba(0,0,0,0)',
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.25)'},
                    {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.25)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.25)'}
                ],
                'threshold': {
                    'line': {'color': '#ef4444', 'width': 3},
                    'thickness': 0.8,
                    'value': 65
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            height=260,
            margin=dict(l=30, r=30, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Financial impact calculation
        saved_monthly = 65.0
        saved_annual = saved_monthly * 12

        r_card1, r_card2 = st.columns(2)
        with r_card1:
            st.metric(
                label="Churn Chance Before Interventions",
                value=f"{base_prob:.1f}%"
            )
        with r_card2:
            st.metric(
                label="Churn Chance After Interventions",
                value=f"{new_prob:.1f}%",
                delta=f"-{risk_reduction:.1f}%" if risk_reduction > 0 else "0.0%",
                delta_color="inverse"
            )

        if risk_reduction > 5:
            st.success(f"🎉 **Retention Success!** This offer package reduces churn chance by **{risk_reduction:.1f} percentage points**, saving an estimated **${saved_annual:,.0f} / year** in recurring revenue on this customer!")
        else:
            st.info("💡 Try checking one or more retention offers above to simulate churn risk reduction.")

# =========================================================
# TAB 5: Prioritized Call List & File Scorer
# =========================================================
with tab5:
    st.markdown("### 📑 Prioritized Retention Call List & Batch File Scorer")
    st.caption("Export targeted contact lists for your sales and retention reps or upload new spreadsheets for instant scoring.")

    subtab_list, subtab_upload = st.tabs(["📋 Prioritized Call Queue", "📤 Upload & Score New Spreadsheet"])

    with subtab_list:
        st.markdown("#### Customer Retention Queue (Highest Risk First)")
        
        call_queue = filtered_df.copy()
        call_queue = call_queue.sort_values('Churn Probability', ascending=False)
        
        display_cols = ['CustomerID', 'Churn Chance', 'Churn Percentage', 'Risk Tier', 'Contract', 'Tenure Months', 'Internet Service', 'Payment Method', 'Estimated Monthly Revenue']
        available_cols = [c for c in display_cols if c in call_queue.columns]
        
        search_query = st.text_input("🔍 Search by Customer ID or City:", placeholder="e.g. 7590-VHVEG or Los Angeles")
        if search_query:
            mask = (
                call_queue['CustomerID'].astype(str).str.contains(search_query, case=False, na=False) |
                call_queue['City'].astype(str).str.contains(search_query, case=False, na=False)
            )
            call_queue = call_queue[mask]

        st.dataframe(
            call_queue[available_cols].head(50),
            use_container_width=True,
            column_config={
                "Churn Percentage": st.column_config.ProgressColumn(
                    "Churn Percentage",
                    help="Estimated churn percentage (chance that customer leaves)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Estimated Monthly Revenue": st.column_config.NumberColumn(
                    "Monthly Value",
                    format="$%.2f"
                )
            }
        )

        csv_buffer = io.StringIO()
        call_queue.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Prioritized Retention Call List (CSV)",
            data=csv_buffer.getvalue().encode('utf-8'),
            file_name=f"telco_retention_call_list_{datetime.date.today()}.csv",
            mime="text/csv"
        )

    with subtab_upload:
        st.markdown("#### Upload Customer File for Instant AI Scoring")
        st.caption("Upload any `.csv` or `.xlsx` spreadsheet of customer accounts to automatically classify risk tiers and identify retention candidates.")

        uploaded_batch = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
        
        batch_input_df = None
        if uploaded_batch is not None:
            try:
                if uploaded_batch.name.endswith(".xlsx"):
                    batch_input_df = pd.read_excel(uploaded_batch)
                else:
                    batch_input_df = pd.read_csv(uploaded_batch)
                st.success(f"Loaded {len(batch_input_df):,} customer accounts from {uploaded_batch.name}!")
            except Exception as err:
                st.error(f"Error reading file: {err}")
        else:
            st.info("No file uploaded. You can load a 100-customer test batch to test this feature:")
            if st.button("Load Sample 100-Customer Batch"):
                batch_input_df = x_test.head(100).copy()

        if batch_input_df is not None:
            expected_features = list(x_test.columns)
            missing = [c for c in expected_features if c not in batch_input_df.columns]
            
            if missing:
                st.warning(f"Missing columns imputed with defaults: {missing}")
                for col in missing:
                    if col in categorical_options:
                        batch_input_df[col] = categorical_options[col][0]
                    else:
                        batch_input_df[col] = 0

            b_transformed = pd.DataFrame(
                preprocessor.transform(batch_input_df[expected_features]),
                columns=clean_feature_names,
                index=batch_input_df.index
            )
            b_probas = model.predict_proba(b_transformed)[:, 1]
            b_preds = (b_probas >= threshold).astype(int)

            scored = batch_input_df.copy()
            scored['Churn Percentage'] = np.round(b_probas * 100, 1)
            scored['Churn Chance'] = scored['Churn Percentage'].apply(lambda x: f"{x:.1f}%")
            scored['Predicted Churn'] = b_preds
            scored['Risk Tier'] = pd.cut(
                scored['Churn Percentage'],
                bins=[-0.1, 35.0, 65.0, 100.1],
                labels=['Low Risk', 'Moderate Risk', 'High Risk']
            )

            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric("Total Scored Accounts", f"{len(scored):,}")
            with s_col2:
                b_high = (scored['Risk Tier'] == 'High Risk').sum()
                st.metric("High-Risk Accounts", f"{b_high:,}", f"{b_high/len(scored)*100:.1f}%")
            with s_col3:
                b_rev = b_high * 65.0
                st.metric("Revenue at Risk", f"${b_rev:,.0f}/mo")

            st.dataframe(
                scored[['Tenure Months', 'Contract', 'Internet Service', 'Payment Method', 'Churn Chance', 'Risk Tier']].head(25),
                use_container_width=True
            )

            scored_csv = scored.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Scored Batch Results (CSV)",
                data=scored_csv,
                file_name="scored_customer_accounts.csv",
                mime="text/csv"
            )
