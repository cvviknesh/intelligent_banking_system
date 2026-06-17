import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from model_trainer import train_and_save

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BankIQ — Intelligent Banking System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main { background: #F5F7FA; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1150px; }

.hero { text-align: center; padding: 2rem 0 1rem; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 3rem; color: #0A2540; margin-bottom: 0.2rem; }
.hero-accent { color: #1A6B6B; }
.hero-sub { font-size: 1rem; color: #6B7C93; font-weight: 300; }

.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 1.5rem 0; }
.kpi { background:#fff; border-radius:14px; padding:1.1rem 1rem; text-align:center; border:1px solid #E8ECF0; }
.kpi-val { font-size:1.8rem; font-weight:600; color:#0A2540; }
.kpi-lbl { font-size:0.78rem; color:#6B7C93; margin-top:3px; }

.card { background:#fff; border-radius:16px; padding:1.6rem; border:1px solid #E8ECF0; margin-bottom:1rem; }
.card-title { font-family:'DM Serif Display',serif; font-size:1.2rem; color:#0A2540; margin-bottom:1rem; }

.result-safe { background:linear-gradient(135deg,#1B5E20,#2E7D32); border-radius:16px; padding:2rem; text-align:center; color:#fff; }
.result-risk { background:linear-gradient(135deg,#B71C1C,#C62828); border-radius:16px; padding:2rem; text-align:center; color:#fff; }
.result-label { font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase; opacity:0.8; }
.result-verdict { font-family:'DM Serif Display',serif; font-size:2.4rem; margin:0.3rem 0; }
.result-prob { font-size:0.95rem; opacity:0.85; }

.seg-card { border-radius:14px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; color:#fff; }
.seg-title { font-size:1.1rem; font-weight:600; margin-bottom:0.3rem; }
.seg-sub { font-size:0.85rem; opacity:0.88; }

.product-chip { display:inline-block; background:#E8F4FD; color:#1565C0; border-radius:20px; padding:4px 12px; font-size:0.82rem; margin:3px 2px; font-weight:500; }

.risk-badge-low  { background:#E8F5E9; color:#1B5E20; border-radius:99px; padding:3px 12px; font-size:0.82rem; font-weight:600; display:inline-block; }
.risk-badge-med  { background:#FFF8E1; color:#E65100; border-radius:99px; padding:3px 12px; font-size:0.82rem; font-weight:600; display:inline-block; }
.risk-badge-high { background:#FFEBEE; color:#B71C1C; border-radius:99px; padding:3px 12px; font-size:0.82rem; font-weight:600; display:inline-block; }

.factor-row { display:flex; justify-content:space-between; align-items:center; padding:6px 0; border-bottom:0.5px solid #F0F0F0; font-size:13px; }
.factor-name { color:#444; }
.factor-val  { font-weight:500; color:#0A2540; }

div[data-testid="stTabs"] button { font-size:0.95rem; font-weight:500; }
label { font-weight:500 !important; color:#0A2540 !important; font-size:0.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "banking_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_and_save(MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

with st.spinner("Loading AI models..."):
    m = load_model()

loan_model   = m["loan_model"]
kmeans       = m["kmeans"]
scaler       = m["scaler"]
le_edu       = m["le_edu"]
le_area      = m["le_area"]
seg_labels   = m["seg_labels"]
products     = m["products"]
stats        = m["overall_stats"]
df_sample    = pd.DataFrame(m["df_sample"])

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">Bank<span class="hero-accent">IQ</span></div>
  <div class="hero-sub">Intelligent Banking System — Loan Default Prediction · Customer Segmentation · Product Recommendation</div>
</div>
""", unsafe_allow_html=True)

# ── KPI bar ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi"><div class="kpi-val">5,000</div><div class="kpi-lbl">Customer records</div></div>
  <div class="kpi"><div class="kpi-val">82%</div><div class="kpi-lbl">Model accuracy</div></div>
  <div class="kpi"><div class="kpi-val">4</div><div class="kpi-lbl">Customer segments</div></div>
  <div class="kpi"><div class="kpi-val">{stats['default_rate']*100:.1f}%</div><div class="kpi-lbl">Avg default rate</div></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Loan Default Predictor",
    "👥  Customer Segmentation",
    "🎁  Product Recommendation",
    "📊  Data Insights"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LOAN DEFAULT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Predict Loan Default Risk")
    st.markdown("Enter customer details to assess loan default probability using a Random Forest model.")

    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="card"><div class="card-title">👤 Customer Profile</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age          = st.number_input("Age", 18, 75, 35)
            income       = st.number_input("Annual Income (₹)", 10000, 500000, 60000, step=5000)
            emp_years    = st.slider("Employment Years", 0, 35, 5)
            dependents   = st.selectbox("Dependents", [0, 1, 2, 3, 4, 5])
        with c2:
            credit_score = st.slider("Credit Score", 300, 850, 700)
            balance      = st.number_input("Account Balance (₹)", 0, 500000, 25000, step=1000)
            existing_loans = st.selectbox("Existing Loans", [0, 1, 2, 3, 4])
            self_employed  = st.selectbox("Self Employed", ["No", "Yes"])
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card"><div class="card-title">🏦 Loan Details</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            loan_amount = st.number_input("Loan Amount (₹)", 10000, 1000000, 200000, step=10000)
            tenure      = st.selectbox("Tenure (months)", [12, 24, 36, 48, 60, 84, 120])
        with c4:
            education    = st.selectbox("Education", ["Graduate", "Post-Graduate", "Under-Graduate", "Doctorate"])
            property_area= st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        st.markdown('</div>', unsafe_allow_html=True)

        predict_loan = st.button("🔍  Assess Default Risk", use_container_width=True, type="primary")

    with col_result:
        if predict_loan:
            edu_enc  = le_edu.transform([education])[0]
            area_enc = le_area.transform([property_area])[0]
            se_enc   = 1 if self_employed == "Yes" else 0

            features = np.array([[age, income, loan_amount, tenure, credit_score,
                                   emp_years, existing_loans, balance,
                                   dependents, edu_enc, area_enc, se_enc]])

            prob_default = loan_model.predict_proba(features)[0][1]
            is_default   = prob_default >= 0.45

            if is_default:
                st.markdown(f"""
                <div class="result-risk">
                  <div class="result-label">Risk Assessment</div>
                  <div class="result-verdict">⚠️ High Risk</div>
                  <div class="result-prob">Default probability: {prob_default*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-safe">
                  <div class="result-label">Risk Assessment</div>
                  <div class="result-verdict">✅ Low Risk</div>
                  <div class="result-prob">Default probability: {prob_default*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Risk factors
            st.markdown('<div class="card"><div class="card-title">📋 Risk Factor Analysis</div>', unsafe_allow_html=True)

            lti   = loan_amount / income
            cs_lbl = "Good ✅" if credit_score >= 700 else ("Fair ⚠️" if credit_score >= 600 else "Poor 🔴")
            lti_lbl= "Low ✅"  if lti < 3 else ("Moderate ⚠️" if lti < 5 else "High 🔴")
            el_lbl = "Safe ✅" if existing_loans <= 1 else ("Caution ⚠️" if existing_loans <= 2 else "Risky 🔴")
            ey_lbl = "Stable ✅" if emp_years >= 3 else "Short ⚠️"

            st.markdown(f"""
            <div class="factor-row"><span class="factor-name">Credit Score ({credit_score})</span><span class="factor-val">{cs_lbl}</span></div>
            <div class="factor-row"><span class="factor-name">Loan-to-Income Ratio ({lti:.1f}x)</span><span class="factor-val">{lti_lbl}</span></div>
            <div class="factor-row"><span class="factor-name">Existing Loans ({existing_loans})</span><span class="factor-val">{el_lbl}</span></div>
            <div class="factor-row"><span class="factor-name">Employment Stability ({emp_years} yrs)</span><span class="factor-val">{ey_lbl}</span></div>
            <div class="factor-row"><span class="factor-name">Account Balance (₹{balance:,})</span><span class="factor-val">{'Healthy ✅' if balance > 20000 else 'Low ⚠️'}</span></div>
            """, unsafe_allow_html=True)

            # Recommendation
            if is_default:
                st.error("💡 **Recommendation:** Decline loan or request additional collateral. Consider offering a smaller loan amount with stricter terms.")
            else:
                st.success("💡 **Recommendation:** Approve loan. Customer shows strong repayment indicators.")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center; padding:3rem 1rem;">
              <div style="font-size:3rem;">🏦</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#0A2540;margin:0.5rem 0;">
                AI-Powered Risk Assessment
              </div>
              <div style="color:#6B7C93;font-size:0.92rem;line-height:1.7;">
                Fill in customer details and click<br><b>Assess Default Risk</b> to get an<br>instant ML prediction.
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CUSTOMER SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Customer Segmentation")
    st.markdown("Enter customer details to identify which segment they belong to using K-Means clustering.")

    col_seg_form, col_seg_result = st.columns([1, 1], gap="large")

    with col_seg_form:
        st.markdown('<div class="card"><div class="card-title">👤 Customer Details</div>', unsafe_allow_html=True)
        s_age    = st.number_input("Age", 18, 75, 40, key="s_age")
        s_income = st.number_input("Annual Income (₹)", 10000, 500000, 65000, step=5000, key="s_income")
        s_balance= st.number_input("Account Balance (₹)", 0, 500000, 30000, step=1000, key="s_balance")
        s_score  = st.slider("Spending Score (1–100)", 1, 100, 55, key="s_score")
        s_loans  = st.selectbox("Existing Loans", [0,1,2,3,4], key="s_loans")
        s_credit = st.slider("Credit Score", 300, 850, 710, key="s_credit")
        st.markdown('</div>', unsafe_allow_html=True)

        segment_btn = st.button("👥  Find Customer Segment", use_container_width=True, type="primary")

    with col_seg_result:
        if segment_btn:
            X_input = scaler.transform([[s_age, s_income, s_balance, s_score, s_loans, s_credit]])
            seg_id  = kmeans.predict(X_input)[0]
            seg_name, seg_emoji, seg_color = seg_labels[seg_id]

            st.markdown(f"""
            <div class="seg-card" style="background:linear-gradient(135deg,{seg_color},{seg_color}CC);">
              <div style="font-size:2.5rem;margin-bottom:0.3rem;">{seg_emoji}</div>
              <div class="seg-title">{seg_name} Customer</div>
              <div class="seg-sub">Segment {seg_id+1} of 4 customer groups</div>
            </div>""", unsafe_allow_html=True)

            # Segment profile
            st.markdown('<div class="card"><div class="card-title">📊 Segment Profile</div>', unsafe_allow_html=True)
            prof = m["seg_profiles"]
            st.markdown(f"""
            <div class="factor-row"><span class="factor-name">Avg Segment Income</span><span class="factor-val">₹{prof['Income'][seg_id]:,.0f}</span></div>
            <div class="factor-row"><span class="factor-name">Avg Spending Score</span><span class="factor-val">{prof['SpendingScore'][seg_id]:.0f}/100</span></div>
            <div class="factor-row"><span class="factor-name">Avg Account Balance</span><span class="factor-val">₹{prof['AccountBalance'][seg_id]:,.0f}</span></div>
            <div class="factor-row"><span class="factor-name">Avg Credit Score</span><span class="factor-val">{prof['CreditScore'][seg_id]:.0f}</span></div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Segment distribution chart
            seg_counts = df_sample["Segment"].value_counts().sort_index()
            seg_names  = [seg_labels[i][0] for i in seg_counts.index]
            seg_colors = [seg_labels[i][2] for i in seg_counts.index]

            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")
            bars = ax.bar(seg_names, seg_counts.values, color=seg_colors, edgecolor="white", linewidth=1.5, width=0.55)
            ax.bar(seg_names[list(seg_counts.index).index(seg_id)],
                   seg_counts.values[list(seg_counts.index).index(seg_id)],
                   color=seg_color, edgecolor="#FFD700", linewidth=2.5, width=0.55, label="You")
            ax.set_title("Segment Distribution", fontsize=11, fontweight="bold", color="#0A2540", pad=10)
            ax.set_ylabel("Customers", fontsize=9, color="#6B7C93")
            ax.tick_params(colors="#6B7C93", labelsize=8)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.yaxis.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 1rem;">
              <div style="font-size:3rem;">👥</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#0A2540;margin:0.5rem 0;">K-Means Segmentation</div>
              <div style="color:#6B7C93;font-size:0.9rem;line-height:1.7;">4 customer segments:<br>
              💎 Premium · 🟢 Stable · 📈 Growth · 🔴 At-Risk</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRODUCT RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Product Recommendation Engine")
    st.markdown("Get personalised banking product recommendations based on customer profile.")

    col_prod_form, col_prod_result = st.columns([1, 1], gap="large")

    with col_prod_form:
        st.markdown('<div class="card"><div class="card-title">👤 Customer Profile</div>', unsafe_allow_html=True)
        p_age    = st.number_input("Age", 18, 75, 38, key="p_age")
        p_income = st.number_input("Annual Income (₹)", 10000, 500000, 75000, step=5000, key="p_income")
        p_balance= st.number_input("Account Balance (₹)", 0, 500000, 45000, step=1000, key="p_balance")
        p_score  = st.slider("Spending Score", 1, 100, 65, key="p_score")
        p_loans  = st.selectbox("Existing Loans", [0,1,2,3,4], key="p_loans")
        p_credit = st.slider("Credit Score", 300, 850, 730, key="p_credit")
        st.markdown('</div>', unsafe_allow_html=True)

        recommend_btn = st.button("🎁  Get Recommendations", use_container_width=True, type="primary")

    with col_prod_result:
        if recommend_btn:
            X_input  = scaler.transform([[p_age, p_income, p_balance, p_score, p_loans, p_credit]])
            seg_id   = kmeans.predict(X_input)[0]
            seg_name, seg_emoji, seg_color = seg_labels[seg_id]
            rec_products = products[seg_name]

            st.markdown(f"""
            <div class="seg-card" style="background:linear-gradient(135deg,{seg_color},{seg_color}CC);margin-bottom:1rem;">
              <div style="font-size:0.8rem;opacity:0.85;letter-spacing:0.1em;text-transform:uppercase;">Customer Segment</div>
              <div style="font-size:1.5rem;font-weight:600;margin:0.2rem 0;">{seg_emoji} {seg_name}</div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="card"><div class="card-title">🎁 Recommended Products</div>', unsafe_allow_html=True)

            icons = ["⭐", "💳", "📈", "🏠"]
            descs = {
                "Platinum Credit Card":   "Exclusive rewards, airport lounge access, high credit limit",
                "Wealth Management":      "Personalised investment advisory & portfolio management",
                "Priority Banking":       "Dedicated relationship manager & premium services",
                "Investment Portfolio":   "Diversified equity & debt investment options",
                "Gold Credit Card":       "Cashback rewards, fuel surcharge waiver, travel benefits",
                "Home Loan":              "Competitive interest rates, flexible repayment options",
                "Mutual Funds":           "SIP starting ₹500/month, diversified fund options",
                "Fixed Deposit":          "Guaranteed returns up to 7.5% p.a.",
                "Personal Loan":          "Instant approval, no collateral, up to ₹5L",
                "Silver Credit Card":     "Basic rewards program, low annual fee",
                "Recurring Deposit":      "Build savings habit, flexible tenure 6–60 months",
                "Insurance Plans":        "Life & health coverage tailored to your needs",
                "Basic Savings Account":  "Zero balance account, free debit card",
                "Financial Counselling":  "Free 1-on-1 session to improve financial health",
                "Secured Credit Card":    "Build credit score with fixed deposit as security",
                "Micro Loans":            "Small loans up to ₹50,000, quick disbursal",
            }

            for i, prod in enumerate(rec_products):
                priority = ["🥇 Best Match", "🥈 Recommended", "🥉 Consider", "💡 Also Available"][i]
                st.markdown(f"""
                <div style="background:#F8FAFC;border-radius:12px;padding:1rem 1.2rem;margin-bottom:0.6rem;border-left:4px solid {seg_color};">
                  <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-weight:600;color:#0A2540;font-size:0.95rem;">{icons[i]}  {prod}</div>
                    <div style="font-size:0.75rem;color:#6B7C93;">{priority}</div>
                  </div>
                  <div style="font-size:0.82rem;color:#6B7C93;margin-top:4px;">{descs.get(prod,'')}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:3rem 1rem;">
              <div style="font-size:3rem;">🎁</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#0A2540;margin:0.5rem 0;">Smart Recommendations</div>
              <div style="color:#6B7C93;font-size:0.9rem;line-height:1.7;">Products are tailored to each<br>customer's segment and financial profile.</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DATA INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Portfolio Analytics Dashboard")

    col_a, col_b = st.columns(2, gap="large")

    # Chart 1 — Default rate by credit score band
    with col_a:
        st.markdown('<div class="card"><div class="card-title">Default Rate by Credit Score Band</div>', unsafe_allow_html=True)
        bins   = [300, 500, 600, 650, 700, 750, 850]
        labels = ["300–500", "500–600", "600–650", "650–700", "700–750", "750–850"]
        df_sample["CS_Band"] = pd.cut(df_sample["CreditScore"], bins=bins, labels=labels)
        cs_default = df_sample.groupby("CS_Band", observed=True)["Default"].mean() * 100

        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        colors = ["#C62828","#E53935","#EF9A9A","#A5D6A7","#66BB6A","#2E7D32"]
        ax.bar(cs_default.index, cs_default.values, color=colors, edgecolor="white", linewidth=1)
        ax.set_xlabel("Credit Score Band", fontsize=9, color="#6B7C93")
        ax.set_ylabel("Default Rate (%)", fontsize=9, color="#6B7C93")
        ax.tick_params(colors="#6B7C93", labelsize=8, rotation=30)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 2 — Income distribution by segment
    with col_b:
        st.markdown('<div class="card"><div class="card-title">Income Distribution by Segment</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        for seg_id in sorted(df_sample["Segment"].unique()):
            seg_name, _, seg_color = seg_labels[seg_id]
            data = df_sample[df_sample["Segment"]==seg_id]["Income"] / 1000
            ax.hist(data, bins=20, alpha=0.65, label=seg_name, color=seg_color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel("Annual Income (₹ thousands)", fontsize=9, color="#6B7C93")
        ax.set_ylabel("Count", fontsize=9, color="#6B7C93")
        ax.legend(fontsize=8, framealpha=0.5)
        ax.tick_params(colors="#6B7C93", labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    col_c, col_d = st.columns(2, gap="large")

    # Chart 3 — Segment pie
    with col_c:
        st.markdown('<div class="card"><div class="card-title">Customer Segment Distribution</div>', unsafe_allow_html=True)
        seg_counts = df_sample["Segment"].value_counts().sort_index()
        seg_names_list  = [seg_labels[i][0] for i in seg_counts.index]
        seg_colors_list = [seg_labels[i][2] for i in seg_counts.index]
        seg_emojis_list = [seg_labels[i][1] for i in seg_counts.index]

        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor("#FFFFFF")
        wedges, texts, autotexts = ax.pie(
            seg_counts.values,
            labels=[f"{e} {n}" for e, n in zip(seg_emojis_list, seg_names_list)],
            autopct="%1.0f%%", colors=seg_colors_list,
            startangle=140, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=2)
        )
        for t in texts:     t.set_fontsize(9)
        for t in autotexts: t.set_fontsize(8); t.set_color("white"); t.set_fontweight("bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # Chart 4 — Loan amount vs income scatter
    with col_d:
        st.markdown('<div class="card"><div class="card-title">Loan Amount vs Income (Default Status)</div>', unsafe_allow_html=True)
        sample = df_sample.sample(200, random_state=1)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")
        for default_val, color, label in [(0,"#2E7D32","No Default"),(1,"#C62828","Default")]:
            mask = sample["Default"] == default_val
            ax.scatter(sample[mask]["Income"]/1000, sample[mask]["LoanAmount"]/1000,
                       c=color, alpha=0.55, s=18, label=label)
        ax.set_xlabel("Income (₹K)", fontsize=9, color="#6B7C93")
        ax.set_ylabel("Loan Amount (₹K)", fontsize=9, color="#6B7C93")
        ax.legend(fontsize=8, framealpha=0.5)
        ax.tick_params(colors="#6B7C93", labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.yaxis.grid(True, alpha=0.3); ax.xaxis.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#9E9E9E;font-size:0.82rem;margin-top:3rem;padding-top:1.5rem;border-top:1px solid #E8ECF0;">
  Built by <strong>Viknesh C</strong> · Data Science Portfolio ·
  <a href="https://github.com/cvviknesh/intelligent_banking_system" style="color:#1A6B6B;text-decoration:none;">GitHub</a> ·
  <a href="https://linkedin.com/in/viknesh01" style="color:#1A6B6B;text-decoration:none;">LinkedIn</a>
</div>
""", unsafe_allow_html=True)
