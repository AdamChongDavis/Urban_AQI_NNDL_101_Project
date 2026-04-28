"""
Linear Regression Baseline for PM2.5 Forecasting
Uses lagged features from a 24-hour sliding window to predict PM2.5 one hour ahead.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# 1. Load all 12 station CSVs
DATA_DIR = "beijing+multi+site+air+quality+data/PRSA_Data_20130301-20170228"
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
print(f"Found {len(files)} station files")

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(df):,}")

# 2. Build a datetime index
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

# 3. Feature engineering
NUMERIC_FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}
df["wd_deg"] = df["wd"].map(WIND_DIR_MAP)
FEATURES = NUMERIC_FEATURES + ["wd_deg"]

# Per-station processing to avoid leaking across stations
WINDOW = 24  # hours of history
TARGET = "PM2.5"

all_X, all_y, all_dt = [], [], []

for station, grp in df.groupby("station"):
    grp = grp.sort_values("datetime").reset_index(drop=True)
    # Fill missing values (linear interpolation within each station)
    grp[FEATURES] = grp[FEATURES].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")
    # Drop rows still missing target
    grp = grp.dropna(subset=[TARGET])

    vals = grp[FEATURES].values # (T, F)
    target = grp[TARGET].values # (T,)
    datetimes = grp["datetime"].values

    # Build sliding-window samples
    for i in range(WINDOW, len(grp) - 1):
        window_flat = vals[i - WINDOW:i].flatten() # 24 * len(FEATURES)
        label = target[i + 1] # 1-hour-ahead PM2.5
        dt = datetimes[i + 1]
        if not np.isnan(window_flat).any() and not np.isnan(label):
            all_X.append(window_flat)
            all_y.append(label)
            all_dt.append(dt)

X = np.array(all_X)
y = np.array(all_y)
dt_arr = np.array(all_dt)
print(f"Total samples: {len(X):,}  |  Features per sample: {X.shape[1]}")

# 4. Temporal train/test split (last ~10 months = test)
SPLIT_DATE = np.datetime64("2016-04-01")
train_mask = dt_arr < SPLIT_DATE
test_mask  = dt_arr >= SPLIT_DATE

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
print(f"Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

# 5. Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# 6. Train Linear Regression 
model = LinearRegression()
model.fit(X_train_s, y_train)

# 7. Evaluate 
y_pred_train = model.predict(X_train_s)
y_pred_test  = model.predict(X_test_s)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae  = mean_absolute_error(y_train, y_pred_train)
test_rmse  = np.sqrt(mean_squared_error(y_test,  y_pred_test))
test_mae   = mean_absolute_error(y_test,  y_pred_test)

# Clamp negatives for interpretability (model may predict <0)
y_pred_test_clipped = np.clip(y_pred_test, 0, None)
test_rmse_clipped = np.sqrt(mean_squared_error(y_test, y_pred_test_clipped))
test_mae_clipped  = mean_absolute_error(y_test, y_pred_test_clipped)

print("\n── Linear Regression Baseline Results ──────────────────────────")
print(f"  Train RMSE : {train_rmse:.2f}  |  Train MAE : {train_mae:.2f}")
print(f"  Test  RMSE : {test_rmse:.2f}  |  Test  MAE : {test_mae:.2f}")
print(f"  Test  RMSE (clipped≥0): {test_rmse_clipped:.2f}  |  MAE: {test_mae_clipped:.2f}")
print(f"  y_test mean: {y_test.mean():.2f}  std: {y_test.std():.2f}")
print(f"  Negative predictions: {(y_pred_test < 0).sum():,} / {len(y_pred_test):,}")
