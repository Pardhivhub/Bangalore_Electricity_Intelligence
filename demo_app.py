"""
Bangalore Electricity Demand Predictor — Advanced Intelligence System
Standalone demo — runs locally without Hopsworks / Comet ML.
Usage: streamlit run demo_app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bangalore Electricity AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Professional CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .app-header {
        font-size: 1.95rem;
        font-weight: 800;
        color: #E8E8E8;
        text-align: left;
        padding: 0.3rem 0 0.1rem 0;
        letter-spacing: -0.5px;
    }
    .app-header span { color: #F7A731; }
    .app-subtitle {
        font-size: 0.95rem;
        color: #8899AA;
        text-align: left;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #D0D0D0;
        border-bottom: 2px solid #263245;
        padding-bottom: 0.35rem;
        margin-bottom: 1.2rem;
        margin-top: 1rem;
    }
    
    /* Cards and Metrics */
    .metric-card {
        background: linear-gradient(135deg, #141E30 0%, #1A2740 100%);
        border-radius: 10px;
        padding: 1.1rem;
        border: 1px solid #263245;
        margin-bottom: 0.6rem;
    }
    .card-value { font-size: 2.2rem; font-weight: 700; color: #F7A731; line-height: 1.1; }
    .card-label { font-size: 0.82rem; color: #8899AA; font-weight: 500; }
    .card-sub { font-size: 0.78rem; color: #667788; }
    
    div[data-testid="stMetric"] {
        background: #141E30;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        border: 1px solid #263245;
    }
    div[data-testid="stMetric"] label { color: #8899AA !important; font-size: 0.85rem !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { font-weight: 700 !important; color: #F7A731; }

    /* Alert Boxes */
    .alert-box {
        background-color: rgba(220, 53, 69, 0.15);
        border-left: 4px solid #dc3545;
        padding: 0.8rem 1rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 1rem;
    }
    .alert-title { color: #ff6b6b; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.2rem; display: flex; align-items: center; gap: 0.5rem; }
    .alert-text { color: #e8e8e8; font-size: 0.85rem; margin: 0; }
    
    .insight-box {
        background-color: rgba(79, 195, 247, 0.1);
        border-left: 4px solid #4FC3F7;
        padding: 0.8rem 1rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 1rem;
    }
    .insight-title { color: #4FC3F7; font-weight: 700; font-size: 0.95rem; margin-bottom: 0.2rem; display: flex; align-items: center; gap: 0.5rem; }
    .insight-text { color: #e8e8e8; font-size: 0.85rem; margin: 0; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid #263245; }
    .stTabs [data-baseweb="tab"] { padding: 0.9rem 1.2rem; font-weight: 500; font-size: 0.9rem; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    .sidebar-label { font-size: 0.8rem; color: #8899AA; font-weight: 500; margin-bottom: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ─── Data Configuration ─────────────────────────────────────────────────
KARNATAKA_REGIONS = {
    0: {"name": "BESCOM North", "full": "Bangalore North", "lat": 13.0358, "lon": 77.5970, "base_mw": 1200},
    1: {"name": "BESCOM South", "full": "Bangalore South", "lat": 12.9000, "lon": 77.5800, "base_mw": 1100},
    2: {"name": "BESCOM East",  "full": "Whitefield / KR Puram", "lat": 12.9698, "lon": 77.7500, "base_mw": 900},
    3: {"name": "BESCOM West",  "full": "Rajajinagar / Peenya", "lat": 12.9900, "lon": 77.5200, "base_mw": 1000},
    4: {"name": "MESCOM",       "full": "Mysore Region", "lat": 12.2958, "lon": 76.6394, "base_mw": 800},
    5: {"name": "CESC",         "full": "Mangalore / Coastal", "lat": 12.9141, "lon": 74.8560, "base_mw": 600},
    6: {"name": "HESCOM",       "full": "Hubli-Dharwad", "lat": 15.3647, "lon": 75.1240, "base_mw": 700},
    7: {"name": "GESCOM",       "full": "Gulbarga / Kalaburagi", "lat": 17.3297, "lon": 76.8343, "base_mw": 500},
}

INDIAN_HOLIDAYS = {
    (1, 26): "Republic Day", (8, 15): "Independence Day", (10, 2): "Gandhi Jayanti", 
    (11, 1): "Karnataka Rajyotsava", (5, 1): "May Day", (12, 25): "Christmas",
}

# Threshold for alerts
OVERLOAD_MULTIPLIER = 1.45  # Alert if demand is 45% above base_mw

# ─── 1. Synthetic Data Generation ──────────────────────────────────────
@st.cache_data(ttl=3600)
def generate_demand_data(days: int = 90):
    np.random.seed(42)
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)
    hours = pd.date_range(start=start, end=now, freq='h', tz='UTC')
    records = []
    
    for hour in hours:
        h, month, dow = hour.hour, hour.month, hour.dayofweek
        if 0 <= h <= 5:      tf = 0.65 + h * 0.02
        elif 6 <= h <= 8:    tf = 0.80 + (h - 6) * 0.08
        elif 9 <= h <= 11:   tf = 1.15 + np.random.uniform(-0.03, 0.05)
        elif 12 <= h <= 14:  tf = 1.05
        elif 15 <= h <= 17:  tf = 1.00
        elif 18 <= h <= 21:  tf = 1.25 + np.random.uniform(-0.05, 0.12)  # Evening peak
        else:                tf = 0.85
        
        season = {3:1.20,4:1.20,5:1.20,6:0.95,7:0.95,8:0.95,11:1.02,12:1.02,1:1.02}.get(month, 1.05)
        wknd = 0.88 if dow >= 5 else 1.0
        
        # Temp varies by time and season
        temp = 24 + 6*np.sin(2*np.pi*(h-14)/24) + np.random.normal(0, 1.5)
        if month in [3,4,5]: temp += 4
        elif month in [6,7,8]: temp -= 3
        
        # In summer, high temp creates an exponential spike in demand (AC usage)
        temp_spike_factor = 1.0 + max(0, (temp - 30) * 0.02)
        
        for code, info in KARNATAKA_REGIONS.items():
            base = info["base_mw"]
            demand = max(100, int(base * tf * season * wknd * temp_spike_factor + np.random.normal(0, base*0.03)))
            
            # Artificial occasional peaks to trigger alerts
            if code == 1 and np.random.random() < 0.02 and h in [19, 20]:
                demand = int(demand * 1.25)
                
            records.append({"date":hour, "sub_region_code":code, "demand":demand,
                            "temperature_2m":round(temp,1)})
    return pd.DataFrame(records)

# ─── 2. Feature Engineering ────────────────────────────────────────────
def create_features(df, n_lags=24):
    all_f, all_t = [], []
    for code in df["sub_region_code"].unique():
        rd = df[df.sub_region_code==code].sort_values("date").reset_index(drop=True)
        for i in range(n_lags, len(rd)-1):
            row = {f"demand_lag_{lag+1}": rd.loc[i-lag-1,"demand"] for lag in range(n_lags)}
            row.update({
                "temperature_2m": rd.loc[i,"temperature_2m"], 
                "hour": rd.loc[i,"date"].hour,
                "day_of_week": rd.loc[i,"date"].dayofweek, 
                "month": rd.loc[i,"date"].month,
                "is_weekend": int(rd.loc[i,"date"].dayofweek>=5),
                "is_holiday": int((rd.loc[i,"date"].month,rd.loc[i,"date"].day) in INDIAN_HOLIDAYS),
                "sub_region_code": code, 
                "avg_demand_24h": np.mean([rd.loc[i-j-1,"demand"] for j in range(24)]),
                "date": rd.loc[i,"date"]  # kept for timeline
            })
            all_f.append(row)
            all_t.append(rd.loc[i+1,"demand"])
    return pd.DataFrame(all_f), pd.Series(all_t, name="target")

# ─── 3. Multi-Model Training & Evaluation ──────────────────────────────
@st.cache_resource
def train_multiple_models(_X, _y, _cols):
    """Train multiple models to demonstrate research depth and academic methodology."""
    models = {
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, max_features=0.8, n_jobs=-1, random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    trained_models = {}
    for name, m in models.items():
        m.fit(_X[_cols], _y)
        trained_models[name] = m
    return trained_models

# ─── 4. Sub-routines for Forecasing & Helpers ──────────────────────────
def est_temp(h, month):
    base = {3:28,4:28,5:28,6:21,7:21,8:21,11:23,12:23,1:23}.get(month, 24)
    return round(base + 6*np.sin(2*np.pi*(h-14)/24), 1)

def forecast_future(model, fcols, raw, fdate, target_hours, temp_modifier=0.0):
    results = []
    for rc in sorted(raw["sub_region_code"].unique()):
        buf = list(raw[raw.sub_region_code==rc].sort_values("date").tail(24)["demand"].values)
        for h in target_hours:
            dt = pd.Timestamp(year=fdate.year,month=fdate.month,day=fdate.day,hour=h,tz='UTC')
            row = {f"demand_lag_{l+1}": buf[-(l+1)] for l in range(24)}
            # Apply what-if temperature modifier
            t = est_temp(h, dt.month) + temp_modifier
            row.update({"temperature_2m":t, "hour":h, "day_of_week":dt.dayofweek,
                        "month":dt.month, "is_weekend":int(dt.dayofweek>=5),
                        "is_holiday":int((dt.month,dt.day) in INDIAN_HOLIDAYS),
                        "avg_demand_24h":np.mean(buf[-24:])})
            
            pred = max(100, round(model.predict(pd.DataFrame([row])[fcols])[0]))
            buf.append(pred)
            results.append({"date":dt,"hour":h,"sub_region_code":rc,
                            "region_name":KARNATAKA_REGIONS[rc]["name"],
                            "region_full":KARNATAKA_REGIONS[rc]["full"],
                            "predicted_demand":pred,"est_temperature":t})
    return pd.DataFrame(results)

def render_alert(title, msg):
    st.markdown(f"""
    <div class="alert-box">
        <div class="alert-title">Critical Alert: {title}</div>
        <p class="alert-text">{msg}</p>
    </div>
    """, unsafe_allow_html=True)

def render_insight(title, msg):
    st.markdown(f"""
    <div class="insight-box">
        <div class="insight-title">System Insight: {title}</div>
        <p class="insight-text">{msg}</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#                            APPLICATION BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="app-header">Bangalore <span>Electricity Intelligence</span> System</div>', unsafe_allow_html=True)
current_date = pd.to_datetime(datetime.utcnow(), utc=True).floor("H")
st.markdown(f'<div class="app-subtitle">AI-driven Decision System & Forecasting for Karnataka Power Grid  ·  {current_date.strftime("%d %b %Y, %H:%M")}</div>', unsafe_allow_html=True)

# ─── Sidebar Data Processing ───
st.sidebar.markdown("### System Configuration")
n_days = st.sidebar.slider("Training Window (days)", 30, 150, 60)

with st.spinner("Initializing Data Lake..."):
    raw_data = generate_demand_data(days=n_days)
with st.spinner("Extracting Temporal Features..."):
    features, targets = create_features(raw_data, n_lags=24)
    cutoff = features["date"].max() - timedelta(hours=48)
    train_mask = features["date"] < cutoff
    X_train, y_train = features[train_mask], targets[train_mask]
    X_test, y_test = features[~train_mask], targets[~train_mask]
    feature_cols = [c for c in features.columns if c not in ["date","sub_region_code"]]

with st.spinner("Training Algorithm Ensemble..."):
    trained_models = train_multiple_models(X_train, y_train, feature_cols)
    primary_model_name = "Gradient Boosting"
    primary_model = trained_models[primary_model_name]
    
    # Calculate Test MAE for primary model
    test_mae = mean_absolute_error(y_test, primary_model.predict(X_test[feature_cols]))

st.sidebar.markdown("---")
st.sidebar.markdown("**Engine Status**")
st.sidebar.caption(f"Status: Online")
st.sidebar.caption(f"Historical Records: {len(raw_data):,}")
st.sidebar.caption(f"Active Algorithm: {primary_model_name}")
st.sidebar.caption(f"Validation MAE: {test_mae:.1f} MW")

# ═══════════════════════════════════════════════════════════════════════
#                              TABS SETUP
# ═══════════════════════════════════════════════════════════════════════
tab_forecast, tab_whatif, tab_monitor, tab_map = st.tabs([
    "Future Forecast & Alerts", 
    "What-If Simulation", 
    "Research & Model Diagnostics", 
    "Live Grid Map"
])

# ═══════════════════ TAB 1: FUTURE FORECAST & INTELLIGENCE ═════════════
with tab_forecast:
    col_input, col_output = st.columns([1, 2.5], gap="large")

    with col_input:
        st.markdown('<div class="section-title">Forecast Parameters</div>', unsafe_allow_html=True)
        today = date.today()
        default_date = today + timedelta(days=1)
        forecast_date = st.date_input("Target Date", value=default_date, min_value=today, max_value=today+timedelta(days=30))
        target_hours = list(range(24))
        predict_btn = st.button("Generate Intelligence Report", type="primary", use_container_width=True)

    with col_output:
        st.markdown('<div class="section-title">AI Grid Intelligence Report</div>', unsafe_allow_html=True)
        
        forecast_df = forecast_future(primary_model, feature_cols, raw_data, forecast_date, target_hours)
        
        # ── Intelligence Layer & Alerts ──
        total_by_hour = forecast_df.groupby("hour")["predicted_demand"].sum()
        peak_hour, peak_val = total_by_hour.idxmax(), total_by_hour.max()
        off_hour, off_val = total_by_hour.idxmin(), total_by_hour.min()
        
        # Check for Overload Alerts
        overload_thresholds = []
        region_utilization = {}
        
        for rc in KARNATAKA_REGIONS.keys():
            r_data = forecast_df[forecast_df.sub_region_code == rc]
            r_max = r_data["predicted_demand"].max()
            r_base = KARNATAKA_REGIONS[rc]["base_mw"]
            
            # Track overall utilization to find the safest region to route power from
            region_utilization[KARNATAKA_REGIONS[rc]["name"]] = r_max / r_base
            
            if r_max > r_base * OVERLOAD_MULTIPLIER:
                max_hour = r_data.loc[r_data["predicted_demand"].idxmax(), "hour"]
                overload_thresholds.append({"region": KARNATAKA_REGIONS[rc]["name"], "hr": max_hour, "val": r_max, "base": r_base, "util": r_max/r_base})
                
        if overload_thresholds:
            # Sort by highest overload
            overload_thresholds = sorted(overload_thresholds, key=lambda x: x["util"], reverse=True)
            # Find the safest region (least utilized)
            safest_region = min(region_utilization, key=region_utilization.get)
            
            # Only show top 2 critical alerts so we don't spam the UI
            for alert in overload_thresholds[:2]:
                render_alert(f"Overload Risk in {alert['region']}", 
                             f"Demand projected to hit {alert['val']:,} MW at {alert['hr']:02d}:00 (Safe Base Load: {alert['base']} MW). Action recommended: Route excess power from {safest_region}.")
            
            if len(overload_thresholds) > 2:
                render_insight("Widespread Grid Stress", f"{len(overload_thresholds)} regions are exhibiting high stress levels. Displaying top 2 critical regions above.")
        else:
            render_insight("Grid Stability", "No major overloads predicted. Grid is balanced under normal operating thresholds.")

        # Rate of increase insight
        morning_jump = total_by_hour.get(9, 0) - total_by_hour.get(5, 0)
        if morning_jump > total_by_hour.mean() * 0.2:
            render_insight("Ramp-up Warning", f"Sharp morning demand increase detected: +{morning_jump:,.0f} MW between 05:00 and 09:00. Ensure peaker plants are active.")

        # ── Metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Projected Peak", f"{peak_val:,} MW", f"Hour {peak_hour:02d}:00")
        m2.metric("Projected Off-Peak", f"{off_val:,} MW", f"Hour {off_hour:02d}:00")
        m3.metric("Daily Average Volume", f"{int(total_by_hour.mean()):,} MW")
        m4.metric("Average System Temp", f"{forecast_df['est_temperature'].mean():.1f} °C")

        # ── Visualization ──
        fig = px.area(
            forecast_df, x="hour", y="predicted_demand", color="region_name",
            title=f"Hourly Grid Demand Forecast — {forecast_date.strftime('%d %b %Y')}",
            template="plotly_dark",
            labels={"predicted_demand":"Demand (MW)", "hour":"Hour of Day", "region_name":"Region"},
            color_discrete_sequence=["#4FC3F7","#FF8A65","#81C784","#BA68C8","#FFD54F","#4DB6AC","#F06292","#90A4AE"],
        )
        fig.update_layout(
            height=400, 
            legend=dict(orientation="h", y=1.08, font=dict(size=11)), 
            xaxis=dict(dtick=2), 
            margin=dict(t=50,b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════ TAB 2: WHAT-IF SIMULATION ═════════════════════════
with tab_whatif:
    st.markdown('<div class="section-title">Scenario Sandbox (What-If Analysis)</div>', unsafe_allow_html=True)
    st.markdown(
        "Simulate external grid conditions and observe real-time demand impacts. "
        "Useful for capacity planning against extreme weather events or sudden behavioral shifts."
    )
    
    scol1, scol2 = st.columns([1, 2.5], gap="large")
    
    with scol1:
        st.markdown('**Adjust Simulation Variables**')
        sim_date = date.today() + timedelta(days=1)
        sim_temp_delta = st.slider("Temperature Anomaly (°C)", min_value=-5.0, max_value=8.0, value=2.0, step=0.5,
                                   help="Simulate a heatwave or cold front. Positive values indicate a systemic heatwave.")
        
        st.info("Baseline: Tomorrow's default forecast vs Adjusted scenario based on parameters above.")
        run_sim_btn = st.button("Execute Simulation", type="primary")

    with scol2:
        # Generate Baseline
        baseline_df = forecast_future(primary_model, feature_cols, raw_data, sim_date, list(range(24)), temp_modifier=0.0)
        # Generate Simulated
        simulated_df = forecast_future(primary_model, feature_cols, raw_data, sim_date, list(range(24)), temp_modifier=sim_temp_delta)
        
        base_total = baseline_df.groupby("hour")["predicted_demand"].sum()
        sim_total = simulated_df.groupby("hour")["predicted_demand"].sum()
        
        diff_total = sim_total.sum() - base_total.sum()
        diff_pct = (diff_total / base_total.sum()) * 100
        
        # Display Impact Insight
        if sim_temp_delta > 0:
            render_insight("Heatwave Impact Analysis", f"A positive offset of {sim_temp_delta}°C results in an estimated load expansion of {diff_pct:.1f}%. Total additional load requirement is projected at {diff_total:,.0f} MWh/day.")
        elif sim_temp_delta < 0:
            render_insight("Cold Front Impact Analysis", f"A negative offset of {sim_temp_delta}°C results in an estimated load contraction of {abs(diff_pct):.1f}%. Total load variance is projected at {abs(diff_total):,.0f} MWh/day.")
        else:
            st.write("Current simulation aligns completely with baseline metrics. Adjust the simulation parameters to observe operational variances.")
        
        # Overlay Plot
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=base_total.index, y=base_total.values, mode='lines', name='Baseline Expected Load', line=dict(color='#8899AA', dash='dash')))
        fig_sim.add_trace(go.Scatter(x=sim_total.index, y=sim_total.values, mode='lines+markers', name='Simulated Distributed Load', line=dict(color='#FF8A65', width=3)))
        
        fig_sim.update_layout(
            title="Grid Load Forecasting: Baseline vs Simulated",
            template="plotly_dark", height=420,
            xaxis=dict(title="Hour of Day", dtick=2),
            yaxis=dict(title="Aggregate Demand (MW)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_sim, use_container_width=True)


# ═══════════════════ TAB 3: ACADEMIC MODEL DIAGNOSTICS ═════════════════
with tab_monitor:
    st.markdown('<div class="section-title">Algorithm Comparison & Feature Explainability</div>', unsafe_allow_html=True)
    st.markdown("This module assesses multiple foundational algorithms to validate system robustness, alongside an analysis of regression feature importance to ensure model transparency.")
    
    col_mod, col_feat = st.columns([1, 1], gap="large")
    
    with col_mod:
        st.markdown("**1. Ensemble Benchmark (Test Set Evaluation)**")
        model_results = []
        for m_name, m_model in trained_models.items():
            preds = m_model.predict(X_test[feature_cols])
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            model_results.append({"Model Structure": m_name, "MAE (MW)": mae, "Variance Score (R²)": r2})
            
        m_df = pd.DataFrame(model_results).sort_values("MAE (MW)")
        
        fig_comp = px.bar(
            m_df, x="Model Structure", y="MAE (MW)", color="MAE (MW)",
            color_continuous_scale=["#81C784","#FF8A65"], text_auto='.1f',
            title="Mean Absolute Error by Algorithm Structure (Lower is Better)"
        )
        fig_comp.update_layout(
            height=320, template="plotly_dark", coloraxis_showscale=False,
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.dataframe(m_df.style.format({"MAE (MW)":"{:.2f}", "Variance Score (R²)":"{:.4f}"}), use_container_width=True)

    with col_feat:
        st.markdown("**2. System Explainability Diagnostics**")
        
        rf_model = trained_models["Random Forest"]
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        
        feat_map = {
            "temperature_2m": "Temperature Offset",
            "hour": "Temporal Sequence (Hour)",
            "avg_demand_24h": "Moving Average (24h)",
            "day_of_week": "Workweek Distribution",
            "month": "Seasonal Multiplier",
            "sub_region_code": "Grid Region Classification",
            "is_weekend": "Weekend Factor",
            "is_holiday": "Holiday/Event Indicator",
            "demand_lag_1": "Immediate Preceding Load (T-1)",
            "demand_lag_2": "Preceding Load (T-2)"
        }
        
        imp_df["Readable"] = imp_df["Feature"].apply(lambda x: feat_map.get(x, "Historical Time-Series Lag"))
        agg_df = imp_df.groupby("Readable")["Importance"].sum().reset_index().sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(
            agg_df.tail(8), x="Importance", y="Readable", orientation='h',
            title="Primary Features Driving Algorithm Assertions",
            template="plotly_dark", color_discrete_sequence=["#4FC3F7"]
        )
        fig_imp.update_layout(
            height=400, margin=dict(l=150),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_imp, use_container_width=True)


# ═══════════════════ TAB 4: LIVE MAP ═══════════════════════════════════
with tab_map:
    st.markdown('<div class="section-title">Live Grid Status Monitor</div>', unsafe_allow_html=True)

    latest = features.groupby("sub_region_code").tail(1).copy()
    latest["predicted_demand"] = primary_model.predict(latest[feature_cols]).round(0).astype(int)

    col_m, col_t = st.columns([3, 2])
    with col_m:
        md = []
        for _, r in latest.iterrows():
            c = int(r["sub_region_code"])
            info = KARNATAKA_REGIONS[c]
            md.append({"name":info["name"],"full_name":info["full"],"lat":info["lat"],
                        "lon":info["lon"],"predicted_demand":int(r["predicted_demand"]),
                        "radius":int(r["predicted_demand"])*3.5})
        mdf = pd.DataFrame(md)
        mn, mx = mdf["predicted_demand"].min(), mdf["predicted_demand"].max()
        def gc(v):
            t=(v-mn)/max(mx-mn,1); return [int(255*t), int(100+155*(1-t)), 50, 200]
        mdf["color"] = mdf["predicted_demand"].apply(gc)

        layer = pdk.Layer("ScatterplotLayer", data=mdf, get_position=["lon","lat"],
                          get_radius="radius", get_fill_color="color", pickable=True, auto_highlight=True)
        view = pdk.ViewState(latitude=14.0, longitude=76.0, zoom=6.2, pitch=45)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
            tooltip={"html":"<b>{name}</b><br/>{full_name}<br/>Current Load: <b>{predicted_demand} MW</b>"},
            map_style="mapbox://styles/mapbox/dark-v10"))

    with col_t:
        st.markdown('<div class="section-title">Current Zonal Metrics</div>', unsafe_allow_html=True)
        total = 0
        for _, r in mdf.sort_values("predicted_demand", ascending=False).iterrows():
            total += r["predicted_demand"]
            c1, c2 = st.columns([3,1])
            c1.markdown(f"**{r['name']}** <br/><span style='color:#8899AA;font-size:0.8rem'>{r['full_name']}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='font-size:1.2rem;font-weight:600;color:#F7A731'>{r['predicted_demand']:,}</span>", unsafe_allow_html=True)
        st.divider()
        st.metric("Total Synchronized Load", f"{total:,} MW")

