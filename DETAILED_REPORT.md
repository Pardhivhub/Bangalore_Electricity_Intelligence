# ⚡ Bangalore Electricity Intelligence System (BEIS)
**Detailed Project & Architectural Integrity Report**

This report provides a comprehensive technical breakdown of the system components, the data physics involved, and the underlying AI modeling architecture developed for the Karnataka power grid.

---

## 1. 📂 Core Architectural Framework

### A. The Master Intelligence Hub (`demo_app.py`)
This is the **primary execution engine**. It consolidates the original multi-file cloud pipeline into a high-performance local forecasting dashboard.
- **Role:** Handles synthetic data extraction, constructs lag-features, trains model ensembles, and renders the 4-tab user interface (Forecasting, Simulation, Research Diagnostics, and Live Grid Monitor).
- **Core Intelligence Layer:**
  - **Dynamic Overload Protection:** Automatically detects if any region (e.g., BESCOM South) hits >145% of its base MW capacity.
  - **Auto-Routing Engine:** If an overload is detected, the system automatically finds the region with the *lowest* instantaneous utilization and recommends a power distribution route (e.g., "Route excess power from CESC").

### B. Feature Construction Hub (`/src/`)
These files originally built for the NYISO (New York) grid were fully recalibrated for the Karnataka context:
1.  **`src/component/data_info.py`**: Redefined geographic coordinates to the Bangalore center point (12.97°N, 77.59°E). Switched the weather API logic from NYC timezones to `Asia/Kolkata`.
2.  **`src/component/model_info.py`**: Engineered the `TemporalFeaturesEngineer`. This module constructs the 24-hour lag features and seasonality markers (hour, day, month). All US-federal holiday logic was replaced with a custom Indian Holiday calendar.
3.  **`src/component/feature_group_config.py`**: Configured the metadata schema, renaming feature groups to `bangalore_kptcl_*` for administrative cataloging.
4.  **`src/frontend.py`**: Rapped the Streamlit map and time-series plotting logic to focus on the 8 main Karnataka regions (including BESCOM North/South/East/West, MESCOM, CESC, HESCOM, and GESCOM).

---

## 2. 📊 Machine Learning Model Ensembles

The system utilizes an **Ensemble Benchmarking** strategy, training three separate models every time it loads to ensure predictive accuracy and research integrity.

### 🔘 Model A: Gradient Boosting Regressor (Primary Engine)
- **Library:** `sklearn.ensemble.GradientBoostingRegressor`
- **Purpose:** Primary forecasting model. Excellent at capturing non-linear relationships, like how demand spikes exponentially when temperatures exceed 30°C due to Air Conditioning.
- **Parameters:**
    - `n_estimators = 150`: Number of boosting stages.
    - `learning_rate = 0.08`: Step-size shrinkage to prevent overfitting.
    - `max_depth = 5`: Limit to tree complexity.
    - `subsample = 0.8`: Fraction of samples used for fitting individual trees, adding stochastic robustness.

### 🔘 Model B: Random Forest Regressor (Diagnostics Engine)
- **Library:** `sklearn.ensemble.RandomForestRegressor`
- **Purpose:** Specifically used for **Feature Explainability**. It generates the "Variables Driving the Prediction" charts to show which factors (Temperature, Hour, or Lagged Load) are moving the grid.
- **Parameters:**
    - `n_estimators = 100`, `max_depth = 10`.

### 🔘 Model C: Linear Regression (Baseline Engine)
- **Library:** `sklearn.linear_model.LinearRegression`
- **Purpose:** Acts as a statistical control group to prove that simple linear models cannot handle the complexity of grid physics as well as modern tree ensembles.

---

## 3. 📡 Data Integrity & Source Physics

### Where does the data come from?
Due to live grid data from KPTCL being restricted, the application utilizes a **High-Calibration Synthetic Grid Generator** that simulates actual Bangalore power physics.

**Key Parameters & Multipliers:**
1.  **Grid Baselines:** Hardcoded base loads for 8 sub-regions (e.g., BESCOM North = 1200 MW, GESCOM = 500 MW).
2.  **Seasonal Physics:**
    - **Summer Spikes (March-May):** +4°C temperature offset + exponential demand multiplier (`AC Usage Factor`).
    - **Monsoon Adjustment (June-August):** -3°C offset + 0.95x load multiplier.
3.  **Human Behavioral Cycles:**
    - **The Indian Peak:** Daily surges at 06:00-09:00 (Morning prep) and 18:00-21:00 (Evening lighting/cooling).
    - **The Weekend Deflation:** A strict `0.88x` multiplier applied to weekends (Saturday/Sunday) to represent the industrial shutdown cycle.
4.  **Lag Features:** Every prediction is fed by the **24 individual MW readings from the past day**, ensuring the model understands "where the grid was" just an hour ago.

---

## 4. 🚀 "Big Project" Competitive Upgrades
These features upgrade the project from a student assignment to a research-grade system:
- ✅ **What-If Sandbox**: Simulate a heatwave (e.g., +4°C Temperature Anomaly) and view the projected grid expansion in real-time.
- ✅ **Intelligence Report**: Automated text-based report parsing, identifying ramp-up warnings and critical failures.
- ✅ **Multi-Region Map**: A standard visual heatmap of the state of Karnataka.
- ✅ **Research Metrics**: Real-time MAE (Mean Absolute Error) and Variance Scores for all models.
