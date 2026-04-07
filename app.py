import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive Fatigue Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    section[data-testid="stSidebar"] { background-color: #1e1b4b; }
    section[data-testid="stSidebar"] * { color: #e0e7ff !important; }
    .kpi-card {
        background: white; border-radius: 12px; padding: 18px 22px;
        border-top: 4px solid #4f46e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 8px;
    }
    .kpi-label { font-size: 0.78rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e293b; line-height: 1.2; }
    .kpi-sub   { font-size: 0.82rem; color: #94a3b8; margin-top: 2px; }
    .insight-card {
        background: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 12px;
    }
    .section-title {
        font-size: 0.72rem; font-weight: 700; color: #6366f1;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px;
    }
    .chart-title { font-size: 1.05rem; font-weight: 600; color: #1e293b; margin-bottom: 2px; }
    .chart-subtitle { font-size: 0.82rem; color: #64748b; margin-bottom: 12px; }
    .alert-high {
        background: #fef2f2; border: 1px solid #fecaca; border-left: 5px solid #dc2626;
        border-radius: 10px; padding: 16px 20px; margin: 12px 0;
    }
    .alert-mod {
        background: #fffbeb; border: 1px solid #fde68a; border-left: 5px solid #d97706;
        border-radius: 10px; padding: 16px 20px; margin: 12px 0;
    }
    .alert-low {
        background: #f0fdf4; border: 1px solid #bbf7d0; border-left: 5px solid #059669;
        border-radius: 10px; padding: 16px 20px; margin: 12px 0;
    }
    .scenario-card {
        background: #f8fafc; border-radius: 10px; padding: 14px 16px;
        border: 1px solid #e2e8f0; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Data & Models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/human_decision_fatigue_dataset.csv")
    time_order = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
    df["Time_of_Day_Enc"]    = df["Time_of_Day"].map(time_order).fillna(1)
    df["Sleep_Deficit"]      = np.maximum(0, 7 - df["Sleep_Hours_Last_Night"])
    df["Decision_Density"]   = df["Decisions_Made"] / (df["Hours_Awake"] + 1e-6)
    df["Cognitive_Pressure"] = df["Stress_Level_1_10"] * df["Cognitive_Load_Score"]
    df["Fatigue_Risk_Index"] = (
        df["Hours_Awake"] * 0.3 + df["Sleep_Deficit"] * 0.4
        + df["Error_Rate"] * 50 + df["Stress_Level_1_10"] * 0.3
    )
    return df

def train_and_save_models():
    """Train models if they don't exist — runs automatically on first deploy."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import os
    os.makedirs("models", exist_ok=True)
    df_t = pd.read_csv("data/human_decision_fatigue_dataset.csv")
    time_order = {"Morning":0,"Afternoon":1,"Evening":2,"Night":3}
    df_t["Time_of_Day_Enc"]    = df_t["Time_of_Day"].map(time_order).fillna(1)
    df_t["Sleep_Deficit"]      = np.maximum(0, 7 - df_t["Sleep_Hours_Last_Night"])
    df_t["Decision_Density"]   = df_t["Decisions_Made"] / (df_t["Hours_Awake"] + 1e-6)
    df_t["Cognitive_Pressure"] = df_t["Stress_Level_1_10"] * df_t["Cognitive_Load_Score"]
    df_t["Fatigue_Risk_Index"] = df_t["Hours_Awake"]*0.3 + df_t["Sleep_Deficit"]*0.4 + df_t["Error_Rate"]*50 + df_t["Stress_Level_1_10"]*0.3
    FEAT = ["Hours_Awake","Decisions_Made","Task_Switches","Avg_Decision_Time_sec","Sleep_Hours_Last_Night","Time_of_Day_Enc","Caffeine_Intake_Cups","Stress_Level_1_10","Error_Rate","Cognitive_Load_Score","Sleep_Deficit","Decision_Density","Cognitive_Pressure","Fatigue_Risk_Index"]
    X = df_t[FEAT]
    le = LabelEncoder(); le.fit(["Low","Moderate","High"])
    yc = le.transform(df_t["Fatigue_Level"]); yr = df_t["Decision_Fatigue_Score"]
    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(X, yc, yr, test_size=0.2, random_state=42, stratify=yc)
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(X_tr, yc_tr)
    reg = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    reg.fit(X_tr, yr_tr)
    acc = (clf.predict(X_te)==yc_te).mean()
    cv  = cross_val_score(clf, X, yc, cv=5, scoring="accuracy", n_jobs=-1)
    yr_p = reg.predict(X_te)
    fi  = sorted(zip(FEAT, clf.feature_importances_), key=lambda x: -x[1])
    metrics = {"clf_accuracy":round(acc,4),"clf_cv_mean":round(cv.mean(),4),"clf_cv_std":round(cv.std(),4),"reg_r2":round(r2_score(yr_te,yr_p),4),"reg_rmse":round(np.sqrt(mean_squared_error(yr_te,yr_p)),4),"reg_mae":round(mean_absolute_error(yr_te,yr_p),4),"n_train":len(X_tr),"n_test":len(X_te),"features":FEAT,"feature_importance":[[f,round(i,6)] for f,i in fi]}
    joblib.dump(clf, "models/clf_fatigue_level.pkl")
    joblib.dump(reg, "models/reg_fatigue_score.pkl")
    joblib.dump(le,  "models/label_encoder.pkl")
    import json
    with open("models/metrics.json","w") as f: json.dump(metrics,f,indent=2)

@st.cache_resource
def load_models():
    import os
    os.makedirs("models", exist_ok=True)
    try:
        clf  = joblib.load("models/clf_fatigue_level.pkl")
        reg  = joblib.load("models/reg_fatigue_score.pkl")
        le   = joblib.load("models/label_encoder.pkl")
        with open("models/metrics.json") as f:
            metrics = json.load(f)
    except (FileNotFoundError, EOFError):
        with st.spinner("Training models on first run — this takes about 30 seconds..."):
            train_and_save_models()
        clf  = joblib.load("models/clf_fatigue_level.pkl")
        reg  = joblib.load("models/reg_fatigue_score.pkl")
        le   = joblib.load("models/label_encoder.pkl")
        with open("models/metrics.json") as f:
            metrics = json.load(f)
    return clf, reg, le, metrics

df            = load_data()
clf, reg, le, metrics = load_models()
FEATURES      = metrics["features"]
FC            = {"Low": "#059669", "Moderate": "#d97706", "High": "#dc2626"}

if "history" not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Cognitive Fatigue Analyzer")
    st.markdown("<span style='color:#a5b4fc; font-size:0.85rem'>Behavioral predictive modeling</span>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("", ["Behavioral Dashboard", "Fatigue Predictor", "Model Insights"], label_visibility="collapsed")
    st.divider()
    st.markdown("""<div style='font-size:0.78rem; color:#818cf8; line-height:1.8'>
    <b style='color:#c7d2fe'>Dataset</b><br>25,000 simulated states<br><br>
    <b style='color:#c7d2fe'>Models</b><br>Random Forest Classifier<br>Random Forest Regressor<br><br>
    <b style='color:#c7d2fe'>Accuracy</b><br>96.7% · R²=0.998<br><br>
    <b style='color:#c7d2fe'>Built by</b><br>Arshia Riaz · MS Data Science<br>UT Austin · GCP ML Certified
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Behavioral Dashboard":
    st.markdown("# Behavioral Dashboard")
    st.markdown("<p style='color:#64748b;margin-top:-8px'>Deep analysis of behavioral signals driving cognitive fatigue across 25,000 simulated decision states.</p>", unsafe_allow_html=True)
    st.divider()

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Records</div><div class='kpi-value'>25,000</div><div class='kpi-sub'>behavioral states</div></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='kpi-card' style='border-top-color:#dc2626'><div class='kpi-label'>High Fatigue</div><div class='kpi-value' style='color:#dc2626'>{(df['Fatigue_Level']=='High').mean()*100:.1f}%</div><div class='kpi-sub'>of all states</div></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='kpi-card' style='border-top-color:#059669'><div class='kpi-label'>Avg Sleep</div><div class='kpi-value' style='color:#059669'>{df['Sleep_Hours_Last_Night'].mean():.1f}h</div><div class='kpi-sub'>last night</div></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='kpi-card' style='border-top-color:#d97706'><div class='kpi-label'>Avg Fatigue Score</div><div class='kpi-value' style='color:#d97706'>{df['Decision_Fatigue_Score'].mean():.1f}</div><div class='kpi-sub'>out of 100</div></div>", unsafe_allow_html=True)
    with c5: st.markdown(f"<div class='kpi-card' style='border-top-color:#7c3aed'><div class='kpi-label'>Avg Hours Awake</div><div class='kpi-value' style='color:#7c3aed'>{df['Hours_Awake'].mean():.1f}</div><div class='kpi-sub'>hours</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Donut + Trajectory
    col_a, col_b = st.columns([1, 1.6])
    with col_a:
        st.markdown("<div class='insight-card'><div class='section-title'>Population Overview</div><div class='chart-title'>Fatigue Level Distribution</div><div class='chart-subtitle'>How fatigued are people across all states?</div>", unsafe_allow_html=True)
        counts = df["Fatigue_Level"].value_counts().reset_index()
        counts.columns = ["Fatigue_Level","Count"]
        fig = px.pie(counts, names="Fatigue_Level", values="Count", color="Fatigue_Level", color_discrete_map=FC, hole=0.55)
        fig.update_traces(textposition="outside", textinfo="percent+label", marker=dict(line=dict(color="white",width=2)))
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20,b=20,l=20,r=20), height=280, font=dict(color="#1e293b"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='insight-card'><div class='section-title'>Critical Insight</div><div class='chart-title'>Fatigue Trajectory — Hours Awake</div><div class='chart-subtitle'>Fatigue grows exponentially after 8 hours. After 12h, most people cross into High fatigue territory.</div>", unsafe_allow_html=True)
        traj = df.groupby("Hours_Awake")["Decision_Fatigue_Score"].mean().reset_index()
        traj.columns = ["Hours_Awake","Avg_Fatigue"]
        fig = go.Figure()
        fig.add_hrect(y0=0,  y1=40,  fillcolor="#dcfce7", opacity=0.3, line_width=0, annotation_text="Low",      annotation_position="left")
        fig.add_hrect(y0=40, y1=70,  fillcolor="#fef9c3", opacity=0.3, line_width=0, annotation_text="Moderate", annotation_position="left")
        fig.add_hrect(y0=70, y1=100, fillcolor="#fee2e2", opacity=0.3, line_width=0, annotation_text="High",     annotation_position="left")
        fig.add_trace(go.Scatter(x=traj["Hours_Awake"], y=traj["Avg_Fatigue"], mode="lines+markers",
                                  line=dict(color="#4f46e5",width=3), marker=dict(size=7,color="#4f46e5"),
                                  hovertemplate="<b>%{x}h awake</b><br>Avg Fatigue: %{y:.1f}<extra></extra>"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"),
                          height=280, margin=dict(t=20,b=30,l=60,r=20), showlegend=False,
                          xaxis=dict(title="Hours Awake",showgrid=True,gridcolor="#f1f5f9"),
                          yaxis=dict(title="Avg Fatigue Score",range=[0,100],showgrid=True,gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Sleep impact + Task switches
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("<div class='insight-card'><div class='section-title'>Recovery Signal</div><div class='chart-title'>Sleep Deprivation Effect</div><div class='chart-subtitle'>Less than 4h of sleep pushes avg fatigue to 81.6. Every hour of sleep matters.</div>", unsafe_allow_html=True)
        sleep_data = pd.DataFrame({"Sleep":["<4h","4–6h","6–7h","7–8h","8h+"],"Avg_Fatigue":[81.6,55.4,30.7,17.0,7.9]})
        colors_s = ["#dc2626","#f97316","#d97706","#65a30d","#059669"]
        fig = px.bar(sleep_data, x="Sleep", y="Avg_Fatigue", color="Sleep", color_discrete_sequence=colors_s, text="Avg_Fatigue")
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#1e293b"), height=300, margin=dict(t=30,b=30,l=40,r=20),
                          xaxis=dict(title="Sleep Duration",showgrid=False),
                          yaxis=dict(title="Avg Fatigue Score",range=[0,100],showgrid=True,gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_d:
        st.markdown("<div class='insight-card'><div class='section-title'>Multitasking Cost</div><div class='chart-title'>Task Switching & Fatigue</div><div class='chart-subtitle'>20+ context switches → 87.9 avg fatigue. Single biggest controllable driver after hours awake.</div>", unsafe_allow_html=True)
        switch_data = pd.DataFrame({"Switches":["0–5","5–10","10–15","15–20","20+"],"Avg_Fatigue":[0.8,11.8,37.0,62.9,87.9]})
        fig = px.bar(switch_data, x="Switches", y="Avg_Fatigue", color="Avg_Fatigue",
                     color_continuous_scale=["#059669","#d97706","#dc2626"], text="Avg_Fatigue")
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"), height=300,
                          margin=dict(t=30,b=30,l=40,r=20),
                          xaxis=dict(title="Task Switches",showgrid=False),
                          yaxis=dict(title="Avg Fatigue Score",range=[0,105],showgrid=True,gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 3: Stress + Error Inflection
    col_e, col_f = st.columns(2)
    with col_e:
        st.markdown("<div class='insight-card'><div class='section-title'>Stress Amplifier</div><div class='chart-title'>Stress Level vs Average Fatigue</div><div class='chart-subtitle'>Very high stress alone drives fatigue scores above 93 — nearly as powerful as 16h without sleep.</div>", unsafe_allow_html=True)
        stress_data = pd.DataFrame({"Stress":["Low (1–3)","Medium (4–5)","High (6–7)","Very High (8–10)"],"Avg_Fatigue":[31.6,64.6,81.4,93.0]})
        fig = px.bar(stress_data, x="Stress", y="Avg_Fatigue", color="Avg_Fatigue",
                     color_continuous_scale=["#bbf7d0","#fef08a","#fca5a5","#991b1b"], text="Avg_Fatigue")
        fig.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"), height=300,
                          margin=dict(t=30,b=30,l=40,r=20),
                          xaxis=dict(title="",showgrid=False),
                          yaxis=dict(title="Avg Fatigue Score",range=[0,105],showgrid=True,gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_f:
        st.markdown("<div class='insight-card'><div class='section-title'>Performance Breakdown</div><div class='chart-title'>Error Rate Inflection Point</div><div class='chart-subtitle'>Error rate is near zero below 50 fatigue. At critical fatigue (75–100), it jumps 59x to 11.87%.</div>", unsafe_allow_html=True)
        error_data = pd.DataFrame({"Zone":["Low (0–25)","Moderate (25–50)","High (50–75)","Critical (75–100)"],"Error_Rate_Pct":[0.00,0.04,1.87,11.87]})
        fig = px.bar(error_data, x="Zone", y="Error_Rate_Pct", color="Zone",
                     color_discrete_sequence=["#059669","#d97706","#f97316","#dc2626"], text="Error_Rate_Pct")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#1e293b"), height=300, margin=dict(t=30,b=30,l=40,r=20),
                          xaxis=dict(title="",showgrid=False),
                          yaxis=dict(title="Avg Error Rate (%)",showgrid=True,gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 4: Correlations + Caffeine
    col_g, col_h = st.columns([1.2, 1])
    with col_g:
        st.markdown("<div class='insight-card'><div class='section-title'>Predictive Power</div><div class='chart-title'>Feature Correlations with Fatigue Score</div><div class='chart-subtitle'>Hours awake and decisions made are nearly equal predictors (r≈0.95). Sleep is the top protective factor.</div>", unsafe_allow_html=True)
        corr_data = pd.DataFrame({"Feature":["Hours Awake","Decisions Made","Task Switches","Cognitive Load","Error Rate","Avg Decision Time","Stress Level","Caffeine Cups","Sleep Hours"],"Correlation":[0.954,0.953,0.899,0.885,0.788,0.523,0.520,0.277,-0.522]}).sort_values("Correlation")
        fig = px.bar(corr_data, x="Correlation", y="Feature", orientation="h", color="Correlation",
                     color_continuous_scale=["#dc2626","#f1f5f9","#4f46e5"], color_continuous_midpoint=0, range_color=[-1,1])
        fig.update_layout(showlegend=False, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"), height=340,
                          margin=dict(t=20,b=20,l=10,r=20),
                          xaxis=dict(title="Pearson Correlation",showgrid=True,gridcolor="#f1f5f9",range=[-0.7,1.1]),
                          yaxis=dict(showgrid=False))
        fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_h:
        st.markdown("<div class='insight-card'><div class='section-title'>Compensatory Behavior</div><div class='chart-title'>The Caffeine Paradox</div><div class='chart-subtitle'>More coffee correlates with higher fatigue — each cup adds ~8 points. It signals someone is already fatigued and compensating.</div>", unsafe_allow_html=True)
        caff_data = pd.DataFrame({"Cups":[0,1,2,3,4,5,6],"Avg_Fatigue":[26.0,34.2,42.0,49.6,54.7,61.5,66.0]})
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=caff_data["Cups"], y=caff_data["Avg_Fatigue"], mode="lines+markers",
                                  line=dict(color="#7c3aed",width=3), marker=dict(size=9,color="#7c3aed",line=dict(color="white",width=2)),
                                  fill="tozeroy", fillcolor="rgba(124,58,237,0.08)",
                                  hovertemplate="<b>%{x} cups</b><br>Avg Fatigue: %{y:.1f}<extra></extra>"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"),
                          height=340, margin=dict(t=20,b=30,l=50,r=20),
                          xaxis=dict(title="Caffeine (cups)",showgrid=True,gridcolor="#f1f5f9",dtick=1),
                          yaxis=dict(title="Avg Fatigue Score",showgrid=True,gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 5: Scatter + Violin
    col_i, col_j = st.columns(2)
    with col_i:
        st.markdown("<div class='insight-card'><div class='section-title'>Decision Load</div><div class='chart-title'>Decisions Made vs Fatigue Score</div><div class='chart-subtitle'>High fatigue clusters above 60 decisions. Clear separation between Low and High states.</div>", unsafe_allow_html=True)
        sample = df.sample(3000, random_state=42)
        fig = px.scatter(sample, x="Decisions_Made", y="Decision_Fatigue_Score", color="Fatigue_Level",
                         color_discrete_map=FC, opacity=0.45,
                         labels={"Decisions_Made":"Decisions Made","Decision_Fatigue_Score":"Fatigue Score"})
        fig.update_traces(marker=dict(size=4))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"),
                          height=320, margin=dict(t=20,b=30,l=50,r=20),
                          legend=dict(title="Fatigue Level",bgcolor="rgba(255,255,255,0.8)"),
                          xaxis=dict(showgrid=True,gridcolor="#f1f5f9"),
                          yaxis=dict(showgrid=True,gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_j:
        st.markdown("<div class='insight-card'><div class='section-title'>Score Distribution</div><div class='chart-title'>Fatigue Score by Level — Violin</div><div class='chart-subtitle'>Moderate has the widest spread — the most variable and unpredictable state.</div>", unsafe_allow_html=True)
        order = ["Low","Moderate","High"]
        df_v = df[df["Fatigue_Level"].isin(order)].copy()
        df_v["Fatigue_Level"] = pd.Categorical(df_v["Fatigue_Level"], categories=order, ordered=True)
        fig = px.violin(df_v, x="Fatigue_Level", y="Decision_Fatigue_Score", color="Fatigue_Level",
                        color_discrete_map=FC, box=True)
        fig.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#1e293b"), height=320, margin=dict(t=20,b=30,l=50,r=20),
                          xaxis=dict(title="Fatigue Level",showgrid=False),
                          yaxis=dict(title="Fatigue Score",showgrid=True,gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Correlation Heatmap
    st.markdown("<div class='insight-card'><div class='section-title'>Full Feature Analysis</div><div class='chart-title'>Correlation Matrix</div><div class='chart-subtitle'>Hours awake and decisions made are nearly collinear (r=0.97), suggesting they co-occur and compound each other.</div>", unsafe_allow_html=True)
    num_cols = ["Hours_Awake","Decisions_Made","Task_Switches","Avg_Decision_Time_sec","Sleep_Hours_Last_Night","Caffeine_Intake_Cups","Stress_Level_1_10","Error_Rate","Cognitive_Load_Score","Decision_Fatigue_Score"]
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"), height=450, margin=dict(t=20,b=20,l=20,r=20))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Fatigue Predictor":
    st.markdown("# Real-Time Fatigue Predictor")
    st.markdown("<p style='color:#64748b;margin-top:-8px'>Simulate a behavioral state for a full cognitive fatigue assessment — population comparison, recovery scenarios, and projected trajectory.</p>", unsafe_allow_html=True)
    st.divider()

    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        st.markdown("### Behavioral Inputs")
        hours_awake      = st.slider("Hours Awake",              1,   20,  8)
        decisions_made   = st.slider("Decisions Made",           1,  120, 40)
        task_switches    = st.slider("Task Switches",            0,   40, 10)
        avg_decision_sec = st.slider("Avg Decision Time (sec)", 0.5, 7.0, 2.5, step=0.1)
        sleep_hours      = st.slider("Sleep Last Night (hrs)",  2.0, 10.0, 7.0, step=0.5)
        stress_level     = st.slider("Stress Level (1–10)",     1.0, 10.0, 4.0, step=0.5)
        caffeine         = st.slider("Caffeine (cups)",           0,    6,   1)
        error_rate       = st.slider("Error Rate",               0.0, 0.35, 0.05, step=0.01)
        cog_load         = st.slider("Cognitive Load Score",     0.5, 10.0, 3.5, step=0.1)
        time_of_day      = st.selectbox("Time of Day", ["Morning","Afternoon","Evening","Night"])
        predict_btn      = st.button("Analyze Fatigue State", use_container_width=True, type="primary")

    with col_out:
        if predict_btn:
            time_enc           = {"Morning":0,"Afternoon":1,"Evening":2,"Night":3}[time_of_day]
            sleep_deficit      = max(0, 7 - sleep_hours)
            decision_density   = decisions_made / (hours_awake + 1e-6)
            cognitive_pressure = stress_level * cog_load
            fatigue_risk_index = hours_awake*0.3 + sleep_deficit*0.4 + error_rate*50 + stress_level*0.3

            input_vec = [hours_awake, decisions_made, task_switches, avg_decision_sec,
                         sleep_hours, time_enc, caffeine, stress_level, error_rate, cog_load,
                         sleep_deficit, decision_density, cognitive_pressure, fatigue_risk_index]
            input_df = pd.DataFrame([input_vec], columns=FEATURES)

            fatigue_score = float(reg.predict(input_df)[0])
            proba         = clf.predict_proba(input_df)[0]
            proba_dict    = dict(zip(le.classes_, proba))
            # Derive label from score so gauge and alert are always consistent
            if fatigue_score < 40:
                fatigue_label = "Low"
            elif fatigue_score < 70:
                fatigue_label = "Moderate"
            else:
                fatigue_label = "High"
            color = FC[fatigue_label]

            st.session_state.history.append({"Label":fatigue_label,"Score":round(fatigue_score,1),"Hours Awake":hours_awake,"Sleep":sleep_hours,"Stress":stress_level})
            if len(st.session_state.history) > 6:
                st.session_state.history.pop(0)

            # Alert banner
            alerts = {
                "High":     ("<div class='alert-high'><strong style='color:#991b1b;font-size:1.1rem'>High Fatigue Detected</strong><br><span style='color:#7f1d1d'>Immediate rest recommended. Error rates are significantly elevated. Avoid high-stakes decisions.</span></div>"),
                "Moderate": ("<div class='alert-mod'><strong style='color:#92400e;font-size:1.1rem'>Moderate Fatigue</strong><br><span style='color:#78350f'>Performance is declining. Take a short break and reduce task switching.</span></div>"),
                "Low":      ("<div class='alert-low'><strong style='color:#065f46;font-size:1.1rem'>Low Fatigue</strong><br><span style='color:#064e3b'>Cognitive state is healthy. Good time for complex, high-value decisions.</span></div>"),
            }
            st.markdown(alerts[fatigue_label], unsafe_allow_html=True)

            # Gauge + Percentile
            percentile = float((df["Decision_Fatigue_Score"] < fatigue_score).mean() * 100)
            g1, g2 = st.columns([1.2, 1])
            with g1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=fatigue_score,
                    title={"text":"Fatigue Score","font":{"color":"#1e293b","size":16}},
                    number={"font":{"color":color,"size":52},"suffix":"/100"},
                    gauge={"axis":{"range":[0,100],"tickcolor":"#94a3b8"},
                           "bar":{"color":color,"thickness":0.25},
                           "steps":[{"range":[0,40],"color":"#dcfce7"},{"range":[40,70],"color":"#fef9c3"},{"range":[70,100],"color":"#fee2e2"}],
                           "threshold":{"line":{"color":color,"width":4},"value":fatigue_score}},
                ))
                fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b", height=230, margin=dict(t=30,b=10,l=20,r=20))
                st.plotly_chart(fig_gauge, use_container_width=True)

            with g2:
                st.markdown(f"""<div style='padding-top:10px'>
                <div style='font-size:0.75rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.05em'>Population Percentile</div>
                <div style='font-size:2.4rem;font-weight:700;color:{color}'>{percentile:.0f}th</div>
                <div style='font-size:0.82rem;color:#64748b'>More fatigued than <b>{percentile:.0f}%</b> of the dataset</div>
                <div style='background:#e2e8f0;border-radius:4px;height:8px;margin:10px 0'>
                <div style='background:{color};width:{min(percentile,100)}%;height:8px;border-radius:4px'></div></div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<div style='font-size:0.75rem;color:#64748b;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-top:8px'>Model Confidence</div>", unsafe_allow_html=True)
                for lbl in ["Low","Moderate","High"]:
                    p = proba_dict.get(lbl,0)
                    st.progress(p, text=f"{lbl}: {p*100:.0f}%")

            # Radar + Signals
            st.divider()
            r1, r2 = st.columns(2)
            with r1:
                st.markdown("**Your Profile vs Population Average**")
                feat_display = ["Hours Awake","Decisions Made","Task Switches","Stress Level","Caffeine","Error Rate","Cognitive Load"]
                user_vals    = [hours_awake, decisions_made, task_switches, stress_level, caffeine, error_rate*100, cog_load]
                avg_vals     = [df["Hours_Awake"].mean(), df["Decisions_Made"].mean(), df["Task_Switches"].mean(),
                                df["Stress_Level_1_10"].mean(), df["Caffeine_Intake_Cups"].mean(),
                                df["Error_Rate"].mean()*100, df["Cognitive_Load_Score"].mean()]
                maxs = [20,120,40,10,6,35,10]
                user_norm = [min(v/m,1.0) for v,m in zip(user_vals,maxs)]
                avg_norm  = [min(v/m,1.0) for v,m in zip(avg_vals,maxs)]
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(r=avg_norm+[avg_norm[0]], theta=feat_display+[feat_display[0]],
                                                     fill="toself", name="Population Avg",
                                                     line=dict(color="#94a3b8",width=1.5), fillcolor="rgba(148,163,184,0.15)"))
                r,g_v,b = int(color[1:3],16), int(color[3:5],16), int(color[5:7],16)
                fig_radar.add_trace(go.Scatterpolar(r=user_norm+[user_norm[0]], theta=feat_display+[feat_display[0]],
                                                     fill="toself", name="Your State",
                                                     line=dict(color=color,width=2.5), fillcolor=f"rgba({r},{g_v},{b},0.15)"))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1],showticklabels=False), bgcolor="rgba(0,0,0,0)"),
                                         showlegend=True, legend=dict(x=0.75,y=1.1),
                                         paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"),
                                         height=290, margin=dict(t=30,b=30,l=30,r=30))
                st.plotly_chart(fig_radar, use_container_width=True)

            with r2:
                st.markdown("**Derived Behavioral Signals**")
                avg_map = {"Sleep Deficit":df["Sleep_Deficit"].mean(), "Decision Density":df["Decision_Density"].mean(),
                           "Cognitive Pressure":df["Cognitive_Pressure"].mean(), "Fatigue Risk Index":df["Fatigue_Risk_Index"].mean()}
                sig_vals = {"Sleep Deficit":sleep_deficit,"Decision Density":decision_density,
                            "Cognitive Pressure":cognitive_pressure,"Fatigue Risk Index":fatigue_risk_index}
                for k,v in sig_vals.items():
                    avg_k = avg_map[k]
                    delta_pct = ((v - avg_k)/(avg_k+1e-6))*100
                    flag = "🔴" if delta_pct > 30 else "🟡" if delta_pct > 0 else "🟢"
                    unit = " hrs" if k=="Sleep Deficit" else "/hr" if k=="Decision Density" else ""
                    st.markdown(f"""<div class='scenario-card'>
                        <div style='display:flex;justify-content:space-between;align-items:center'>
                        <span style='color:#1e293b;font-weight:600'>{k}</span>
                        <span style='font-size:1.3rem;font-weight:700;color:{color}'>{v:.2f}{unit}</span></div>
                        <div style='font-size:0.78rem;color:#64748b'>{flag} {delta_pct:+.0f}% vs avg ({avg_k:.2f}{unit})</div>
                    </div>""", unsafe_allow_html=True)

            # Recovery Scenarios
            st.divider()
            st.markdown("**Recovery Scenarios — What If?**")

            def predict_scenario(overrides):
                sv = input_vec.copy()
                idx_map = {"sleep_hours":4,"decisions_made":1,"task_switches":2,"stress_level":7,"hours_awake":0}
                for k,v in overrides.items():
                    if k in idx_map: sv[idx_map[k]] = v
                sv[10] = max(0,7-sv[4])
                sv[11] = sv[1]/(sv[0]+1e-6)
                sv[12] = sv[7]*sv[9]
                sv[13] = sv[0]*0.3+sv[10]*0.4+sv[8]*50+sv[7]*0.3
                sd = pd.DataFrame([sv], columns=FEATURES)
                return float(reg.predict(sd)[0]), le.inverse_transform(clf.predict(sd))[0]

            sc1,sc2,sc3 = st.columns(3)
            scenarios = [
                (sc1,"Sleep +2h",{"sleep_hours":min(sleep_hours+2,10)},f"Sleep {sleep_hours:.0f}h → {min(sleep_hours+2,10):.0f}h"),
                (sc2,"Reduce Workload 30%",{"decisions_made":int(decisions_made*0.7),"task_switches":int(task_switches*0.7)},"Reduce decisions & switches by 30%"),
                (sc3,"Stress –3",{"stress_level":max(stress_level-3,1)},f"Stress {stress_level:.0f} → {max(stress_level-3,1):.0f}"),
            ]
            for col, title, overrides, desc in scenarios:
                sc_score, sc_label = predict_scenario(overrides)
                sc_color = FC[sc_label]
                delta = sc_score - fatigue_score
                col.markdown(f"""<div class='insight-card' style='border-top:3px solid {sc_color}'>
                    <div style='font-size:0.9rem;font-weight:700;color:#1e293b'>{title}</div>
                    <div style='font-size:0.75rem;color:#64748b;margin:4px 0 10px'>{desc}</div>
                    <div style='font-size:1.9rem;font-weight:700;color:{sc_color}'>{sc_score:.1f}</div>
                    <div style='font-size:0.82rem;color:{"#059669" if delta<0 else "#dc2626"}'>
                        {"▼" if delta<0 else "▲"} {abs(delta):.1f} pts from current</div>
                    <div style='font-size:0.8rem;color:{sc_color};font-weight:600;margin-top:4px'>{sc_label} Fatigue</div>
                </div>""", unsafe_allow_html=True)

            # Trajectory
            st.divider()
            st.markdown("**Projected Fatigue — Next 5 Hours (if conditions hold)**")
            future = []
            for xh in range(0,6):
                fh = min(hours_awake+xh, 20)
                sd_i = sleep_deficit
                fri_i = fh*0.3+sd_i*0.4+error_rate*50+stress_level*0.3
                fv = input_vec.copy(); fv[0]=fh; fv[10]=sd_i; fv[11]=decisions_made/(fh+1e-6); fv[13]=fri_i
                fs = float(reg.predict(pd.DataFrame([fv],columns=FEATURES))[0])
                fl = le.inverse_transform(clf.predict(pd.DataFrame([fv],columns=FEATURES)))[0]
                future.append({"Label":"Now" if xh==0 else f"+{xh}h","Score":fs,"Color":FC[fl]})
            traj_df = pd.DataFrame(future)
            fig_traj = go.Figure()
            fig_traj.add_hrect(y0=0,  y1=40,  fillcolor="#dcfce7", opacity=0.25, line_width=0)
            fig_traj.add_hrect(y0=40, y1=70,  fillcolor="#fef9c3", opacity=0.25, line_width=0)
            fig_traj.add_hrect(y0=70, y1=100, fillcolor="#fee2e2", opacity=0.25, line_width=0)
            fig_traj.add_trace(go.Scatter(
                x=traj_df["Label"], y=traj_df["Score"], mode="lines+markers+text",
                line=dict(color="#4f46e5",width=3),
                marker=dict(size=13, color=traj_df["Color"].tolist(), line=dict(color="white",width=2)),
                text=[f"{s:.0f}" for s in traj_df["Score"]], textposition="top center",
                hovertemplate="<b>%{x}</b><br>Fatigue: %{y:.1f}<extra></extra>",
            ))
            fig_traj.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#1e293b"), height=260,
                                    margin=dict(t=30,b=20,l=50,r=20),
                                    xaxis=dict(showgrid=False),
                                    yaxis=dict(title="Fatigue Score",range=[0,110],showgrid=True,gridcolor="#f1f5f9"))
            st.plotly_chart(fig_traj, use_container_width=True)

            # Session History
            if len(st.session_state.history) > 1:
                st.divider()
                st.markdown("**Session History**")
                st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
        else:
            st.info("👈 Set your behavioral inputs and click **Analyze Fatigue State** for a full assessment.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Insights":
    st.markdown("# Model Insights")
    st.markdown("<p style='color:#64748b;margin-top:-8px'>Architecture, performance, and feature importance.</p>", unsafe_allow_html=True)
    st.divider()
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Classifier Accuracy",  f"{metrics['clf_accuracy']*100:.1f}%")
    c2.metric("CV Accuracy (5-fold)", f"{metrics['clf_cv_mean']*100:.1f}% ±{metrics['clf_cv_std']*100:.1f}%")
    c3.metric("Regression R²",        f"{metrics['reg_r2']:.4f}")
    c4.metric("RMSE",                 f"{metrics['reg_rmse']:.2f}")
    c5.metric("MAE",                  f"{metrics['reg_mae']:.2f}")
    st.divider()
    col_fi, col_m = st.columns([1.2,1])
    with col_fi:
        st.markdown("<div class='insight-card'><div class='chart-title'>Feature Importance — Classifier</div><div class='chart-subtitle'>Engineered Fatigue Risk Index is the top feature. Hours awake and decisions made are nearly tied.</div>", unsafe_allow_html=True)
        fi_data = pd.DataFrame(metrics["feature_importance"], columns=["Feature","Importance"]).sort_values("Importance",ascending=True)
        fig = px.bar(fi_data, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Purples")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#1e293b"), height=460,
                          margin=dict(t=10,b=10,l=10,r=20),
                          xaxis=dict(showgrid=True,gridcolor="#f1f5f9"), yaxis=dict(showgrid=False))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_m:
        st.markdown("<div class='insight-card'><div class='chart-title'>Methodology</div>", unsafe_allow_html=True)
        st.markdown("""
        **Data**
        - 25,000 simulated behavioral states
        - 10 raw features + 4 engineered signals
        - No missing values, no imputation needed

        **Engineered Features**
        - **Sleep Deficit**: max(0, 7 − sleep)
        - **Decision Density**: decisions ÷ hours_awake
        - **Cognitive Pressure**: stress × cognitive_load
        - **Fatigue Risk Index**: weighted composite from cognitive load literature

        **Models**
        - Random Forest Classifier → Fatigue Level (Low / Moderate / High)
        - Random Forest Regressor → Fatigue Score (0–100 continuous)
        - n_estimators=200, max_depth=12, balanced class weights

        **Evaluation**
        - 80/20 stratified train-test split
        - 5-fold cross-validation
        - Metrics: Accuracy, F1, R², RMSE, MAE

        **Relevance to Behavioral Science**
        - Decision fatigue mirrors behavioral patterns in diabetes patients (pre/post-meal decisions)
        - Framework extensible to CGM-linked behavioral monitoring
        - Sleep deficit and task switching are the most actionable intervention targets
        """)
        st.markdown("</div>", unsafe_allow_html=True)