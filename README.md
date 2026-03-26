# ⚡ Bangalore Electricity Intelligence System (BEIS)
**AI-driven Decision System & Forecasting for Karnataka Power Grid**

---

## 🏛️ Project Architecture
This project is an advanced electricity demand forecasting and grid management system designed specifically for the state of **Karnataka, India**. Transferred from a New York-based (NYISO) architectural template, it has been fully rebuilt into a professional, research-grade dashboard for local utility optimization.

### 🧠 Core Intelligence Layer
Unlike basic prediction models, the BEIS features an **Active Intelligence Layer**:
- **Critical Overload Alerts:** Monitors Zonal loads and triggers alerts when demand exceeds 145% of base capacity.
- **Auto-Routing Engine:** Dynamically calculates the least-stressed region in the grid and recommends power transfer routes to prevent blackouts.
- **Ramp-Up Diagnostics:** Identifies sharp morning demand surges (06:00-09:00) requiring peaker plant synchronization.

---

## 🚀 "Big Project" Features

### 🌡️ What-If Scenario Simulation
A real-time sandbox allowing capacity planners to test grid resilience:
- **Temperature Anomaly Input:** Inject heatwaves (+4°C or more) to see the projected load expansion in MWh/day.
- **Impact Comparison:** Visualization of Baseline vs. Simulated load distribution.

### 📊 Ensemble Model Diagnostics
Under the hood, the system benchmarks three models simultaneously:
1. **Gradient Boosting (GBDT):** Primary non-linear production engine (`n_estimators=150`, `learning_rate=0.08`).
2. **Random Forest:** Used for **System Explainability** through regression feature importance mapping.
3. **Linear Regression:** Statistical control group to validate ensemble performance gains.

---

## 📂 File-by-File Technical Guide

| File | Purpose | Data Source / Engine |
| :--- | :--- | :--- |
| `demo_app.py` | **Master Controller** | GBDT Engine + Streamlit UI |
| `DETAILED_REPORT.md` | **Full Tech Spec** | System parameters & model hyperparams |
| `src/component/data_info.py` | **Grid Config** | Bangalore (12.97°N, 77.59°E) metadata |
| `src/component/model_info.py` | **Feature Engineering** | Indian Holiday logic & 24h Lag features |
| `src/frontend.py` | **Legacy UI Hub** | PyDeck Geocentric Mapping logic |

---

## 🛠️ Installation & Execution
```bash
# Install dependencies
pip install streamlit pandas numpy scikit-learn plotly pydeck

# Run the Intelligence Dashboard
streamlit run demo_app.py
```

---

## 📊 Data & Academic Metrics
- **Data Source:** High-Calibration Synthetic Bangalore Grid Generator (Mimicking actual KPTCL seasonal physics).
- **Validation MAE:** ~27.7 MW (Tested on Hold-out Ensemble).
- **Optimization:** Dynamic routing logic using instantaneous utilization coefficients.
