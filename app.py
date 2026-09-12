import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import torch
import shap
import matplotlib.pyplot as plt

# Streamlit Page Config
st.set_page_config(
    page_title="FinTech Loan Default Prediction System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics (Dark Glassmorphism UI)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    .risk-badge-low {
        background-color: #059669;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .risk-badge-medium {
        background-color: #d97706;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .risk-badge-high {
        background-color: #dc2626;
        color: white;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper: Load Artifacts
@st.cache_resource
def load_models_and_artifacts():
    model_path = os.path.join('models', 'xgboost.joblib')
    scaler_path = os.path.join('models', 'scaler.joblib')
    cols_path = os.path.join('models', 'feature_cols.joblib')
    
    xgb_model = joblib.load(model_path) if os.path.exists(model_path) else None
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    feature_cols = joblib.load(cols_path) if os.path.exists(cols_path) else None
    
    return xgb_model, scaler, feature_cols

xgb_model, scaler, feature_cols = load_models_and_artifacts()

# App Header
st.title("💳 FinTech Loan Default Prediction & Risk Management System")
st.markdown("##### *Production-Grade Risk Engine with Machine Learning, PyTorch DL, & SHAP Explainability*")
st.divider()

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Applicant Risk Assessment",
    "📁 Batch Loan Evaluation",
    "📊 EDA & Financial Analytics",
    "📈 Model Comparison & Benchmark"
])

# ---------------------------------------------------------
# TAB 1: APPLICANT RISK ASSESSMENT
# ---------------------------------------------------------
with tab1:
    st.subheader("Interactive Applicant Credit Decisioning")
    
    col_input, col_result = st.columns([1.1, 1.2])
    
    with col_input:
        st.markdown("### 📋 Loan Application Inputs")
        
        with st.expander("👤 Applicant Demographics & Financials", expanded=True):
            age = st.slider("Applicant Age", 18, 75, 34)
            annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=65000, step=2500)
            emp_length_years = st.slider("Employment History (Years)", 0.0, 30.0, 5.0, step=0.5)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])
            verification_status = st.selectbox("Income Verification", ["Verified", "Source Verified", "Not Verified"])
            
        with st.expander("💳 Credit Bureau Profile", expanded=True):
            credit_score = st.slider("FICO Credit Score", 300, 850, 680)
            dti = st.slider("Debt-to-Income Ratio (DTI %)", 0.0, 65.0, 18.5, step=0.5)
            revol_bal = st.number_input("Revolving Credit Balance ($)", min_value=0, max_value=200000, value=12000, step=1000)
            total_credit_lines = st.slider("Total Credit Lines", 1, 60, 15)
            open_credit_lines = st.slider("Open Active Credit Lines", 1, total_credit_lines, 8)
            public_derogatory = st.selectbox("Public Derogatory Records", [0, 1, 2, 3, 4])
            inquiries_last_6m = st.selectbox("Hard Credit Inquiries (Last 6 Months)", [0, 1, 2, 3, 4, 5, 6])
            
        with st.expander("💵 Requested Loan Terms", expanded=True):
            loan_amount = st.number_input("Requested Principal ($)", min_value=1000, max_value=50000, value=15000, step=500)
            interest_rate = st.slider("Interest Rate (APR %)", 5.0, 32.0, 14.5, step=0.25)
            term_months = st.radio("Loan Term (Months)", [36, 60], horizontal=True)
            loan_intent = st.selectbox("Loan Purpose", [
                "PERSONAL", "DEBT_CONSOLIDATION", "BUSINESS", "MEDICAL", "EDUCATION", "HOME_IMPROVEMENT"
            ])
            
        predict_btn = st.button("🚀 Calculate Default Risk Score", type="primary", use_container_width=True)

    with col_result:
        st.markdown("### 📊 Real-Time Risk Assessment Output")
        
        if predict_btn or 'prediction_done' not in st.session_state:
            st.session_state['prediction_done'] = True
            
            # Construct Raw Input DataFrame
            input_dict = {
                'age': age,
                'emp_length_years': emp_length_years,
                'annual_income': annual_income,
                'credit_score': credit_score,
                'dti': dti,
                'revol_bal': revol_bal,
                'total_credit_lines': total_credit_lines,
                'open_credit_lines': open_credit_lines,
                'public_derogatory': public_derogatory,
                'inquiries_last_6m': inquiries_last_6m,
                'loan_amount': loan_amount,
                'interest_rate': interest_rate,
                'term_months_36': 1 if term_months == 36 else 0,
                'term_months_60': 1 if term_months == 60 else 0,
                'home_ownership_MORTGAGE': 1 if home_ownership == 'MORTGAGE' else 0,
                'home_ownership_OTHER': 1 if home_ownership == 'OTHER' else 0,
                'home_ownership_OWN': 1 if home_ownership == 'OWN' else 0,
                'home_ownership_RENT': 1 if home_ownership == 'RENT' else 0,
                'verification_status_Not Verified': 1 if verification_status == 'Not Verified' else 0,
                'verification_status_Source Verified': 1 if verification_status == 'Source Verified' else 0,
                'verification_status_Verified': 1 if verification_status == 'Verified' else 0,
                'loan_intent_BUSINESS': 1 if loan_intent == 'BUSINESS' else 0,
                'loan_intent_DEBT_CONSOLIDATION': 1 if loan_intent == 'DEBT_CONSOLIDATION' else 0,
                'loan_intent_EDUCATION': 1 if loan_intent == 'EDUCATION' else 0,
                'loan_intent_HOME_IMPROVEMENT': 1 if loan_intent == 'HOME_IMPROVEMENT' else 0,
                'loan_intent_MEDICAL': 1 if loan_intent == 'MEDICAL' else 0,
                'loan_intent_PERSONAL': 1 if loan_intent == 'PERSONAL' else 0
            }
            
            # Engineer Financial Ratios
            input_dict['lti'] = loan_amount / (annual_income + 1.0)
            r = (interest_rate / 100.0) / 12.0
            n = term_months
            monthly_payment = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)
            input_dict['pti'] = monthly_payment / ((annual_income / 12.0) + 1.0)
            input_dict['revol_util_ratio'] = revol_bal / (annual_income + 1.0)
            input_dict['interest_risk_index'] = (interest_rate * dti) / (credit_score + 1.0)
            input_dict['debt_burden_index'] = dti * (loan_amount / 10000.0)
            
            # Format according to feature_cols
            df_input = pd.DataFrame([input_dict])
            
            # Align columns
            for col in feature_cols:
                if col not in df_input.columns:
                    df_input[col] = 0
            df_input = df_input[feature_cols]
            
            # Scale
            X_input_scaled = scaler.transform(df_input)
            
            # Inference
            prob_default = float(xgb_model.predict_proba(X_input_scaled)[0, 1])
            prob_pct = prob_default * 100.0
            
            # Risk Category
            if prob_pct < 30.0:
                risk_cat = "LOW RISK"
                badge_class = "risk-badge-low"
                action_text = "✅ AUTO-APPROVE: Applicant meets high creditworthiness standards."
            elif prob_pct < 70.0:
                risk_cat = "MEDIUM RISK"
                badge_class = "risk-badge-medium"
                action_text = "⚠️ MANUAL REVIEW: Underwriter verification required for debt ratios."
            else:
                risk_cat = "HIGH RISK"
                badge_class = "risk-badge-high"
                action_text = "❌ AUTO-DECLINE: Default probability exceeds risk appetite tolerance."
                
            # Display Score Card
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin:0; color:#94a3b8;">Predicted Default Probability</h4>
                        <h1 style="margin:5px 0; font-size:3rem;">{prob_pct:.1f}%</h1>
                    </div>
                    <div>
                        <span class="{badge_class}">{risk_cat}</span>
                    </div>
                </div>
                <hr style="border: 1px solid rgba(255,255,255,0.1); margin:15px 0;">
                <p style="margin:0; font-size:1.05rem;"><strong>Recommendation:</strong> {action_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress Gauge
            st.progress(min(prob_default, 1.0))
            
            # SHAP Local Explanation
            st.markdown("#### 🔍 Explainable AI (SHAP Driver Analysis)")
            explainer = shap.TreeExplainer(xgb_model)
            df_scaled = pd.DataFrame(X_input_scaled, columns=feature_cols)
            shap_vals = explainer(df_scaled)[0].values
            
            top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
            
            shap_drivers = []
            for idx in top_idx:
                fname = feature_cols[idx]
                sval = shap_vals[idx]
                impact = "🔴 Elevates Default Risk" if sval > 0 else "🟢 Lowers Default Risk"
                shap_drivers.append({"Feature": fname, "SHAP Impact Score": round(sval, 4), "Effect": impact})
                
            st.dataframe(pd.DataFrame(shap_drivers), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: BATCH EVALUATION
# ---------------------------------------------------------
with tab2:
    st.subheader("📁 Batch Loan Portfolio Risk Evaluator")
    st.markdown("Upload applicant CSV file or use the synthetic test dataset to generate bulk credit risk scores.")
    
    if st.button("📥 Load Sample Test Batch (750 Applicants)"):
        test_df = pd.read_csv('data/processed/test.csv')
        st.session_state['batch_df'] = test_df
        
    if 'batch_df' in st.session_state:
        df_batch = st.session_state['batch_df']
        st.write(f"Loaded Batch Dataset: `{df_batch.shape[0]}` rows")
        
        # Run inference
        X_b = df_batch[feature_cols]
        X_b_scaled = scaler.transform(X_b)
        probs = xgb_model.predict_proba(X_b_scaled)[:, 1]
        
        res_df = df_batch.copy()
        res_df['default_probability_%'] = np.round(probs * 100.0, 2)
        res_df['risk_category'] = res_df['default_probability_%'].apply(
            lambda p: 'LOW' if p < 30 else ('MEDIUM' if p < 70 else 'HIGH')
        )
        
        st.dataframe(res_df[['loan_amount', 'annual_income', 'credit_score', 'dti', 'default_probability_%', 'risk_category']], use_container_width=True)
        
        # Category breakdown plot
        cat_counts = res_df['risk_category'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        cat_counts.plot(kind='bar', color=['#2ecc71', '#f39c12', '#e74c3c'], ax=ax)
        ax.set_title("Batch Risk Category Breakdown", fontweight='bold')
        ax.set_ylabel("Applicant Count")
        plt.tight_layout()
        st.pyplot(fig)

# ---------------------------------------------------------
# TAB 3: EDA & FINANCIAL ANALYTICS
# ---------------------------------------------------------
with tab3:
    st.subheader("📊 Exploratory Data Analysis & Behavioral Insights")
    
    eda_dir = 'artifacts/eda_plots'
    if os.path.exists(eda_dir):
        c1, c2 = st.columns(2)
        with c1:
            st.image(os.path.join(eda_dir, 'class_distribution.png'), caption="Loan Default Class Distribution")
            st.image(os.path.join(eda_dir, 'bivariate_insights.png'), caption="Financial Features vs Default Status")
        with c2:
            st.image(os.path.join(eda_dir, 'univariate_distributions.png'), caption="Feature Univariate Distributions")
            st.image(os.path.join(eda_dir, 'correlation_heatmap.png'), caption="Financial Correlation Matrix")

# ---------------------------------------------------------
# TAB 4: MODEL COMPARISON & BENCHMARK
# ---------------------------------------------------------
with tab4:
    st.subheader("📈 Model Benchmarking & Performance Comparison")
    
    comp_json = 'artifacts/model_comparison.json'
    roc_img = 'artifacts/evaluation_plots/roc_comparison.png'
    
    if os.path.exists(comp_json):
        with open(comp_json, 'r') as f:
            metrics_dict = json.load(f)
            
        st.markdown("### Side-by-Side Model Evaluation Table")
        st.dataframe(pd.DataFrame(metrics_dict).T, use_container_width=True)
        
    if os.path.exists(roc_img):
        st.markdown("### Comparative ROC Curves (Test Set)")
        st.image(roc_img, use_container_width=True)
