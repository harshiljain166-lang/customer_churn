import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnIQ · Predictive Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Background ── */
  .stApp {
    background: #0a0d14;
    color: #e2e8f0;
  }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: #0f1420 !important;
    border-right: 1px solid #1e2535;
  }
  section[data-testid="stSidebar"] * { color: #cbd5e0 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label { color: #94a3b8 !important; font-size: 0.78rem !important; }

  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: #111827;
    border: 1px solid #1e2d45;
    border-radius: 14px;
    padding: 20px 24px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s;
  }
  div[data-testid="metric-container"]:hover { transform: translateY(-2px); }
  div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.72rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif; font-size: 2rem !important; color: #f1f5f9 !important; }
  div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

  /* ── Headers ── */
  h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e2535;
    gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-family: 'Syne', sans-serif;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    padding: 10px 20px;
    border-radius: 8px 8px 0 0;
    border: none !important;
  }
  .stTabs [aria-selected="true"] {
    background: #1e2535 !important;
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
  }

  /* ── Cards / Containers ── */
  .card {
    background: #111827;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .card-accent {
    background: linear-gradient(135deg, #0f2744 0%, #0c1a2e 100%);
    border: 1px solid #1e4070;
    border-radius: 16px;
    padding: 24px;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.55rem 1.8rem !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.25) !important;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
  }

  /* ── Inputs ── */
  .stSelectbox > div, .stSlider > div { color: #cbd5e0; }
  .stSlider [data-baseweb="slider"] { margin-top: 4px; }
  div[data-testid="stNumberInput"] input,
  div[data-testid="stTextInput"] input { background: #1a2235 !important; border: 1px solid #2d3a50 !important; color: #e2e8f0 !important; border-radius: 8px !important; }

  /* ── Divider ── */
  hr { border-color: #1e2535; }

  /* ── Plotly chart background fix ── */
  .js-plotly-plot .plotly .bg { fill: transparent !important; }

  /* ── Risk badge ── */
  .risk-high { background: #450a0a; border: 1px solid #b91c1c; border-radius: 8px; padding: 12px 16px; color: #fca5a5; }
  .risk-med  { background: #431407; border: 1px solid #c2410c; border-radius: 8px; padding: 12px 16px; color: #fdba74; }
  .risk-low  { background: #052e16; border: 1px solid #15803d; border-radius: 8px; padding: 12px 16px; color: #86efac; }
  .risk-label { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_prediction_dataset.csv")
    return df

@st.cache_resource
def train_model(df):
    """Train a fresh logistic regression since the pickle may have version issues."""
    data = df.copy()
    data['Churn_bin'] = (data['Churn'] == 'Yes').astype(int)

    cat_cols = ['gender','Partner','Dependents','PhoneService','MultipleLines',
                'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection',
                'TechSupport','StreamingTV','StreamingMovies','Contract',
                'PaperlessBilling','PaymentMethod']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    features = ['gender','SeniorCitizen','Partner','Dependents','tenure',
                'PhoneService','MultipleLines','InternetService','OnlineSecurity',
                'OnlineBackup','DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies','Contract','PaperlessBilling','PaymentMethod',
                'MonthlyCharges','TotalCharges']

    X = data[features]
    y = data['Churn_bin']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model, encoders, features, X_test, y_test

df = load_data()
model, encoders, feature_names, X_test, y_test = train_model(df)

CHART_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='DM Sans', color='#94a3b8', size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', tickfont=dict(color='#64748b')),
    yaxis=dict(gridcolor='#1e2535', linecolor='#1e2535', tickfont=dict(color='#64748b')),
    legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#94a3b8'))
)

PALETTE = ['#38bdf8','#6366f1','#f472b6','#34d399','#fb923c','#a78bfa']


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 20px'>
      <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:800;
                  background:linear-gradient(90deg,#38bdf8,#6366f1);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
        ⚡ ChurnIQ
      </div>
      <div style='font-size:0.72rem;color:#475569;margin-top:4px'>Predictive Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔍 Dataset Filters")
    contract_filter = st.multiselect(
        "Contract Type", df['Contract'].unique(), default=list(df['Contract'].unique())
    )
    internet_filter = st.multiselect(
        "Internet Service", df['InternetService'].unique(), default=list(df['InternetService'].unique())
    )
    tenure_range = st.slider("Tenure (months)", 1, 72, (1, 72))
    charges_range = st.slider("Monthly Charges ($)", float(df['MonthlyCharges'].min()),
                              float(df['MonthlyCharges'].max()),
                              (float(df['MonthlyCharges'].min()), float(df['MonthlyCharges'].max())))

    st.markdown("---")
    st.markdown("<div style='font-size:0.7rem;color:#334155'>© 2025 ChurnIQ Analytics</div>", unsafe_allow_html=True)

# Apply filters
filtered = df[
    df['Contract'].isin(contract_filter) &
    df['InternetService'].isin(internet_filter) &
    df['tenure'].between(*tenure_range) &
    df['MonthlyCharges'].between(*charges_range)
]


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:32px 0 12px'>
  <div style='font-family:Syne,sans-serif;font-size:2.4rem;font-weight:800;
              background:linear-gradient(90deg,#e2e8f0 30%,#38bdf8 70%);
              -webkit-background-clip:text;-webkit-text-fill-color:transparent;
              line-height:1.1'>
    Customer Churn Intelligence
  </div>
  <div style='color:#475569;font-size:0.9rem;margin-top:6px'>
    Real-time risk scoring · Behavioral analytics · Retention modelling
  </div>
</div>
""", unsafe_allow_html=True)

# ─── KPI ROW ──────────────────────────────────────────────────────────────────
churn_rate = (filtered['Churn'] == 'Yes').mean() * 100
avg_tenure  = filtered['tenure'].mean()
avg_charges = filtered['MonthlyCharges'].mean()
total_customers = len(filtered)
at_risk = (filtered['Churn'] == 'Yes').sum()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Customers",  f"{total_customers:,}")
c2.metric("Churn Rate",       f"{churn_rate:.1f}%",   delta=f"{churn_rate - 46.3:.1f}% vs baseline", delta_color="inverse")
c3.metric("At-Risk Customers",f"{at_risk:,}",          delta=f"-{total_customers - at_risk} retained")
c4.metric("Avg Tenure",       f"{avg_tenure:.1f} mo",  delta="months")
c5.metric("Avg Monthly Bill", f"${avg_charges:.2f}",   delta=f"${avg_charges - 67.2:.2f} vs avg", delta_color="inverse")

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊  OVERVIEW", "🔬  DEEP ANALYSIS", "🤖  PREDICT CHURN", "📈  MODEL METRICS"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════
with tab1:
    row1_l, row1_r = st.columns([1, 1])

    with row1_l:
        # Churn Donut
        churn_counts = filtered['Churn'].value_counts()
        fig_donut = go.Figure(go.Pie(
            labels=churn_counts.index,
            values=churn_counts.values,
            hole=0.65,
            marker_colors=['#6366f1','#38bdf8'],
            textfont=dict(color='white', size=13),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'
        ))
        fig_donut.add_annotation(
            text=f"<b>{churn_rate:.0f}%</b><br><span style='font-size:11px'>churn</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=20, color='#f1f5f9'),
            align='center'
        )
        fig_donut.update_layout(title='Churn Distribution', **CHART_LAYOUT,
                                 showlegend=True, height=300)
        st.plotly_chart(fig_donut, use_container_width=True)

    with row1_r:
        # Tenure Histogram
        fig_ten = px.histogram(
            filtered, x='tenure', color='Churn',
            barmode='overlay', nbins=30,
            color_discrete_map={'Yes': '#f472b6', 'No': '#38bdf8'},
            labels={'tenure': 'Tenure (months)', 'count': 'Customers'}
        )
        fig_ten.update_traces(opacity=0.8)
        fig_ten.update_layout(title='Tenure Distribution by Churn', **CHART_LAYOUT, height=300,
                              bargap=0.05)
        st.plotly_chart(fig_ten, use_container_width=True)

    row2_l, row2_r = st.columns([1, 1])

    with row2_l:
        # Monthly Charges Box
        fig_box = px.box(
            filtered, x='Contract', y='MonthlyCharges', color='Churn',
            color_discrete_map={'Yes': '#f472b6', 'No': '#34d399'},
            points='outliers'
        )
        fig_box.update_layout(title='Monthly Charges by Contract & Churn', **CHART_LAYOUT, height=320)
        st.plotly_chart(fig_box, use_container_width=True)

    with row2_r:
        # Internet + Churn stacked bar
        grp = filtered.groupby(['InternetService', 'Churn']).size().reset_index(name='count')
        fig_bar = px.bar(
            grp, x='InternetService', y='count', color='Churn',
            barmode='group',
            color_discrete_map={'Yes': '#f472b6', 'No': '#38bdf8'},
            text='count'
        )
        fig_bar.update_traces(textposition='outside', textfont=dict(color='#94a3b8', size=11))
        fig_bar.update_layout(title='Churn by Internet Service', **CHART_LAYOUT, height=320)
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 2 — DEEP ANALYSIS
# ══════════════════════════════════════════════════════════════════
with tab2:
    # Heatmap — churn rate by contract × payment
    pivot = filtered.groupby(['Contract', 'PaymentMethod']).apply(
        lambda x: (x['Churn'] == 'Yes').mean() * 100).reset_index(name='ChurnRate')
    pivot_table = pivot.pivot(index='Contract', columns='PaymentMethod', values='ChurnRate')

    fig_heat = go.Figure(go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale=[[0,'#0f2744'],[0.5,'#6366f1'],[1,'#f472b6']],
        text=[[f"{v:.1f}%" for v in row] for row in pivot_table.values],
        texttemplate='%{text}',
        hovertemplate='Contract: %{y}<br>Payment: %{x}<br>Churn: %{z:.1f}%<extra></extra>'
    ))
    fig_heat.update_layout(title='Churn Rate Heatmap · Contract × Payment Method',
                           **CHART_LAYOUT, height=280)
    st.plotly_chart(fig_heat, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        # Scatter: tenure vs monthly charges
        fig_sc = px.scatter(
            filtered, x='tenure', y='MonthlyCharges', color='Churn',
            size='TotalCharges', size_max=18,
            color_discrete_map={'Yes': '#f472b6', 'No': '#38bdf8'},
            opacity=0.7,
            hover_data=['customerID', 'Contract']
        )
        fig_sc.update_layout(title='Tenure vs Charges (bubble = Total Spend)',
                              **CHART_LAYOUT, height=340)
        st.plotly_chart(fig_sc, use_container_width=True)

    with col_b:
        # Feature importance (model coefficients)
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': model.coef_[0]
        }).sort_values('Coefficient')
        colors_coef = ['#f472b6' if c > 0 else '#38bdf8' for c in coef_df['Coefficient']]
        fig_coef = go.Figure(go.Bar(
            x=coef_df['Coefficient'], y=coef_df['Feature'],
            orientation='h',
            marker_color=colors_coef,
        ))
        fig_coef.update_layout(title='Model Feature Importance (Coefficients)',
                                **CHART_LAYOUT, height=340)
        st.plotly_chart(fig_coef, use_container_width=True)

    # Service adoption sunburst
    service_cols = ['PhoneService','InternetService','OnlineSecurity','TechSupport','StreamingTV']
    churn_svc = filtered[['Churn'] + service_cols].copy()
    churn_yes = churn_svc[churn_svc['Churn'] == 'Yes']
    svc_counts = {col: (churn_yes[col] == 'Yes').sum() for col in service_cols}
    fig_radar = go.Figure(go.Scatterpolar(
        r=list(svc_counts.values()),
        theta=service_cols,
        fill='toself',
        line_color='#6366f1',
        fillcolor='rgba(99,102,241,0.25)',
        name='Churned'
    ))
    not_churn = churn_svc[churn_svc['Churn'] == 'No']
    svc_counts2 = {col: (not_churn[col] == 'Yes').sum() for col in service_cols}
    fig_radar.add_trace(go.Scatterpolar(
        r=list(svc_counts2.values()),
        theta=service_cols,
        fill='toself',
        line_color='#38bdf8',
        fillcolor='rgba(56,189,248,0.15)',
        name='Retained'
    ))
    fig_radar.update_layout(
        polar=dict(bgcolor='rgba(0,0,0,0)',
                   radialaxis=dict(visible=True, gridcolor='#1e2535', color='#475569'),
                   angularaxis=dict(gridcolor='#1e2535', color='#94a3b8')),
        title='Service Adoption · Churned vs Retained',
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ['xaxis','yaxis']},
        height=380
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════
# TAB 3 — PREDICT CHURN
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:600;
                color:#94a3b8;margin-bottom:20px'>
      Enter customer profile to score churn probability in real-time
    </div>
    """, unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns(3)

    with col_p1:
        st.markdown("**Demographics**")
        p_gender       = st.selectbox("Gender", ["Male","Female"])
        p_senior       = st.selectbox("Senior Citizen", ["No","Yes"])
        p_partner      = st.selectbox("Partner", ["Yes","No"])
        p_dependents   = st.selectbox("Dependents", ["Yes","No"])
        p_tenure       = st.slider("Tenure (months)", 1, 72, 24)

    with col_p2:
        st.markdown("**Services**")
        p_phone        = st.selectbox("Phone Service", ["Yes","No"])
        p_multiline    = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
        p_internet     = st.selectbox("Internet Service", ["Fiber optic","DSL","No"])
        p_security     = st.selectbox("Online Security", ["Yes","No","No internet service"])
        p_backup       = st.selectbox("Online Backup", ["Yes","No","No internet service"])
        p_device       = st.selectbox("Device Protection", ["Yes","No","No internet service"])
        p_tech         = st.selectbox("Tech Support", ["Yes","No","No internet service"])
        p_tv           = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
        p_movies       = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

    with col_p3:
        st.markdown("**Billing**")
        p_contract     = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        p_paperless    = st.selectbox("Paperless Billing", ["Yes","No"])
        p_payment      = st.selectbox("Payment Method", ["Electronic check","Credit card","Bank transfer","Mailed check"])
        p_monthly      = st.slider("Monthly Charges ($)", 18.0, 120.0, 67.0, step=0.5)
        p_total        = st.number_input("Total Charges ($)", min_value=0.0, value=float(p_monthly * p_tenure), step=10.0)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("⚡  Run Churn Prediction", use_container_width=False):
        raw = {
            'gender': p_gender, 'SeniorCitizen': 1 if p_senior == 'Yes' else 0,
            'Partner': p_partner, 'Dependents': p_dependents, 'tenure': p_tenure,
            'PhoneService': p_phone, 'MultipleLines': p_multiline, 'InternetService': p_internet,
            'OnlineSecurity': p_security, 'OnlineBackup': p_backup, 'DeviceProtection': p_device,
            'TechSupport': p_tech, 'StreamingTV': p_tv, 'StreamingMovies': p_movies,
            'Contract': p_contract, 'PaperlessBilling': p_paperless, 'PaymentMethod': p_payment,
            'MonthlyCharges': p_monthly, 'TotalCharges': p_total
        }

        row = pd.DataFrame([raw])
        cat_cols_pred = [c for c in encoders]
        for col in cat_cols_pred:
            le = encoders[col]
            val = row[col].iloc[0]
            if val in le.classes_:
                row[col] = le.transform([val])[0]
            else:
                row[col] = 0

        X_pred = row[feature_names]
        prob = model.predict_proba(X_pred)[0][1]
        pct  = prob * 100

        # Risk bucket
        if pct >= 65:
            risk_cls, risk_label, risk_icon = 'risk-high', 'HIGH RISK', '🔴'
        elif pct >= 40:
            risk_cls, risk_label, risk_icon = 'risk-med',  'MEDIUM RISK','🟡'
        else:
            risk_cls, risk_label, risk_icon = 'risk-low',  'LOW RISK',   '🟢'

        r1, r2 = st.columns([1, 2])
        with r1:
            st.markdown(f"""
            <div class='{risk_cls}' style='text-align:center;padding:24px'>
              <div style='font-size:2.4rem'>{risk_icon}</div>
              <div class='risk-label'>{risk_label}</div>
              <div style='font-size:3rem;font-family:Syne,sans-serif;font-weight:800;margin:8px 0'>
                {pct:.1f}%
              </div>
              <div style='font-size:0.8rem;opacity:0.7'>churn probability</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pct,
                number={'suffix': '%', 'font': {'size': 36, 'family': 'Syne', 'color': '#f1f5f9'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickfont': {'color': '#64748b'}},
                    'bar': {'color': '#f472b6' if pct >= 65 else '#fb923c' if pct >= 40 else '#34d399',
                            'thickness': 0.3},
                    'bgcolor': '#111827',
                    'steps': [
                        {'range': [0, 40],   'color': '#052e16'},
                        {'range': [40, 65],  'color': '#431407'},
                        {'range': [65, 100], 'color': '#450a0a'},
                    ],
                    'threshold': {'line': {'color': 'white', 'width': 2}, 'thickness': 0.7, 'value': pct}
                }
            ))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=240,
                                    font=dict(family='DM Sans', color='#94a3b8'),
                                    margin=dict(l=30, r=30, t=20, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Recommendations
        st.markdown("#### 💡 Retention Recommendations")
        recs = []
        if p_contract == 'Month-to-month':
            recs.append("📋 **Upgrade to annual contract** – offer a 10% discount incentive")
        if p_internet == 'Fiber optic' and p_security == 'No':
            recs.append("🔒 **Bundle Online Security** – free 3-month trial to increase stickiness")
        if p_tenure < 12:
            recs.append("🎁 **Early loyalty reward** – complimentary service upgrade at 12-month mark")
        if p_monthly > 80:
            recs.append("💰 **Review billing plan** – consider a discounted bundle to reduce cost pressure")
        if p_payment == 'Electronic check':
            recs.append("🏦 **Switch to auto-pay** – offer $5/month credit for bank transfer enrollment")
        if not recs:
            recs.append("✅ **Customer appears stable** – continue standard engagement cadence")

        for rec in recs:
            st.markdown(f"> {rec}")


# ══════════════════════════════════════════════════════════════════
# TAB 4 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════
with tab4:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{acc*100:.1f}%")
    m2.metric("Precision", f"{prec*100:.1f}%")
    m3.metric("Recall",    f"{rec*100:.1f}%")
    m4.metric("F1 Score",  f"{f1*100:.1f}%")

    col_roc, col_cm = st.columns(2)

    with col_roc:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode='lines',
            name=f'AUC = {roc_auc:.3f}',
            line=dict(color='#38bdf8', width=2.5),
            fill='tozeroy', fillcolor='rgba(56,189,248,0.1)'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode='lines',
            line=dict(dash='dash', color='#475569', width=1),
            name='Random'
        ))
        fig_roc.update_layout(title=f'ROC Curve  (AUC = {roc_auc:.3f})',
                               xaxis_title='False Positive Rate',
                               yaxis_title='True Positive Rate',
                               **CHART_LAYOUT, height=340)
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=['Predicted No','Predicted Yes'],
            y=['Actual No','Actual Yes'],
            colorscale=[[0,'#0a0d14'],[1,'#6366f1']],
            text=cm, texttemplate='<b>%{text}</b>',
            hovertemplate='%{y} → %{x}: %{z}<extra></extra>'
        ))
        fig_cm.update_layout(title='Confusion Matrix', **CHART_LAYOUT, height=340)
        st.plotly_chart(fig_cm, use_container_width=True)

    # Probability distribution
    prob_df = pd.DataFrame({'prob': y_proba, 'true': y_test.values})
    fig_prob = px.histogram(
        prob_df, x='prob', color=prob_df['true'].map({0:'No Churn',1:'Churned'}),
        barmode='overlay', nbins=30, opacity=0.75,
        color_discrete_map={'Churned':'#f472b6','No Churn':'#38bdf8'},
        labels={'prob':'Predicted Churn Probability','color':'Outcome'}
    )
    fig_prob.add_vline(x=0.5, line_dash='dash', line_color='white', opacity=0.5)
    fig_prob.update_layout(title='Predicted Probability Distribution', **CHART_LAYOUT, height=280)
    st.plotly_chart(fig_prob, use_container_width=True)



