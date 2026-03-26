import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import FunctionTransformer

import xgboost as xgb
import lightgbm as lgb


# ── Indian Public Holidays ──────────────────────────────────────────────
# Since pandas doesn't have an IndianHolidayCalendar built-in,
# we define major Indian national and Karnataka state holidays.
INDIAN_HOLIDAYS = [
    # National holidays (recurring annually – we generate for multiple years)
    (1, 26),   # Republic Day
    (3, 29),   # Holi (approximate, varies by year)
    (4, 14),   # Ambedkar Jayanti / Tamil New Year
    (5, 1),    # May Day / Karnataka Rajyotsava
    (8, 15),   # Independence Day
    (10, 2),   # Gandhi Jayanti
    (11, 1),   # Karnataka Rajyotsava (state holiday)
    (11, 14),  # Children's Day
    (12, 25),  # Christmas
]

def get_indian_holidays(start_date, end_date):
    """Generate a list of Indian holiday dates between start_date and end_date."""
    holidays = []
    for year in range(start_date.year, end_date.year + 1):
        for month, day in INDIAN_HOLIDAYS:
            try:
                holidays.append(pd.Timestamp(year=year, month=month, day=day, tz='UTC'))
            except ValueError:
                pass  # Skip invalid dates (e.g., Feb 29 in non-leap years)
        # Diwali (approximate dates - varies each year)
        diwali_dates = {
            2023: (11, 12), 2024: (11, 1), 2025: (10, 20),
            2026: (11, 8), 2027: (10, 29), 2028: (10, 17),
        }
        if year in diwali_dates:
            m, d = diwali_dates[year]
            holidays.append(pd.Timestamp(year=year, month=m, day=d, tz='UTC'))
    return holidays


def average_demand_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Adds one column with the average demand from
    - 7 days ago
    - 14 days ago
    - 21 days ago
    - 28 days ago
    """
    X['average_demand_last_4_weeks'] = 0.25*(
    X[f'demand_previous_{7*24}_hour'] + \
    X[f'demand_previous_{2*7*24}_hour'] + \
    X[f'demand_previous_{3*7*24}_hour'] + \
    X[f'demand_previous_{4*7*24}_hour']
    )
    return X


class TemporalFeaturesEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn data transformation that adds temporal features:
    - hour
    - day_of_week
    - month
    - is_weekend
    - is_holiday (Indian holidays)
    and removes the `date` datetime column.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        
        X_ = X.copy()
        X_['date']= pd.to_datetime(X_['date'], utc=True)
        
        # Generate numeric columns from datetime
        X_["hour"] = X_['date'].dt.hour
        X_["day_of_week"] = X_['date'].dt.dayofweek
        X_['month'] = X_['date'].dt.month
        X_['is_weekend'] = X_['day_of_week'].isin([5, 6]).astype(int)

        # Use Indian holidays instead of US Federal holidays
        indian_holidays = get_indian_holidays(X_['date'].min(), X_['date'].max())
        X_['is_holiday'] = X_['date'].dt.date.isin(
            [h.date() for h in indian_holidays]
        ).astype(int)
        
        return X_.drop(columns=['date'])

def get_pipeline(**hyperparams) -> Pipeline:

    # sklearn transform
    add_feature_average_demand_last_4_weeks = FunctionTransformer(
        average_demand_last_4_weeks, validate=False)
    
    # sklearn transform
    add_temporal_features = TemporalFeaturesEngineer()

    # sklearn pipeline
    return make_pipeline(
        add_feature_average_demand_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**hyperparams)
    )



def evaluate_model(y_test, y_pred):
    test_mae = mean_absolute_error(y_test, y_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_pred)
    return f"MAE is {test_mae:.4f} and MAPE is: {test_mape:.4f}"
