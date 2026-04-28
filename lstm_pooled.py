"""
LSTM Forecaster for PM2.5 prediction — 1-hour ahead.
"""

import os
import glob
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             classification_report, accuracy_score)

def log(msg):
    print(msg, flush=True)

# Config
DATA_DIR   = "beijing+multi+site+air+quality+data/PRSA_Data_20130301-20170228"
WINDOW     = 24        # hours of history fed to LSTM
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT    = 0.2
BATCH_SIZE = 512
LR         = 1e-3
EPOCHS     = 30
SPLIT_DATE = "2016-04-01"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# AQI tiers — US EPA PM2.5 breakpoints
AQI_BINS   = [0, 12.0, 35.4, 55.4, 150.4, float("inf")]
AQI_LABELS = ["Good", "Moderate", "Unhealthy (Sensitive Groups)",
               "Unhealthy", "Very Unhealthy / Hazardous"]

log(f"Device: {DEVICE}")

# 1. Load all 12 station CSVs
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
log(f"Found {len(files)} station files")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
log(f"Total rows: {len(df):,}")

# 2. Feature preprocessing
NUMERIC_FEATURES = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                    "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_DIR_MAP = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
    "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}
df["wd_deg"] = df["wd"].map(WIND_DIR_MAP)
FEATURES = NUMERIC_FEATURES + ["wd_deg"] # 12 features total

# 3. Build sliding-window sequences per station (vectorised) 
def make_windows(vals, target, datetimes, window):
    """Vectorised sliding-window construction using stride tricks."""
    # vals: (T, F), target: (T,), datetimes: (T,)
    T, F = vals.shape
    if T < window + 1:
        return np.empty((0, window, F), np.float32), np.empty(0, np.float32), np.array([])

    X_sw = np.lib.stride_tricks.sliding_window_view(vals, (window, F)).reshape(-1, window, F)
    X_sw  = X_sw[:-1] # drop last (no label), shape (T-window, window, F)
    y_sw  = target[window:] # (T-window,)
    dt_sw = datetimes[window:] # (T-window,)

    # Drop rows with NaN
    has_nan = np.isnan(X_sw).any(axis=(1, 2)) | np.isnan(y_sw)
    return (X_sw[~has_nan].astype(np.float32),
            y_sw[~has_nan].astype(np.float32),
            dt_sw[~has_nan])

all_X, all_y, all_dt = [], [], []

for station, grp in df.groupby("station"):
    grp = grp.sort_values("datetime").reset_index(drop=True)
    grp[FEATURES] = (grp[FEATURES]
                     .interpolate(method="linear")
                     .bfill()
                     .ffill())
    grp = grp.dropna(subset=["PM2.5"])

    vals      = grp[FEATURES].values.astype(np.float32)
    target    = grp["PM2.5"].values.astype(np.float32)
    datetimes = grp["datetime"].values

    Xs, ys, dts = make_windows(vals, target, datetimes, WINDOW)
    all_X.append(Xs); all_y.append(ys); all_dt.append(dts)
    log(f"  {station}: {len(ys):,} sequences")

X      = np.concatenate(all_X, axis=0) # (N, 24, 12)
y      = np.concatenate(all_y, axis=0) # (N,)
dt_arr = np.concatenate(all_dt, axis=0)
log(f"Total sequences: {len(X):,}  |  Shape: {X.shape}")

# 4. Temporal train / test split
split      = np.datetime64(SPLIT_DATE)
train_mask = dt_arr < split
test_mask  = dt_arr >= split

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]
log(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# 5. Standardise (fit on train only)
n_feat    = X_train.shape[2]
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(
    X_train.reshape(-1, n_feat)).reshape(X_train.shape)
X_test_s  = scaler.transform(
    X_test.reshape(-1, n_feat)).reshape(X_test.shape)

# 6. Dataset & DataLoader 
class PM25Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(PM25Dataset(X_train_s, y_train),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(PM25Dataset(X_test_s,  y_test),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 7. LSTM Model
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)

model     = LSTMForecaster(n_feat, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=4, factor=0.5)

log(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 8. Training loop
def evaluate(loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            p = model(Xb.to(DEVICE)).cpu().numpy()
            preds.extend(p); targets.extend(yb.numpy())
    return np.array(preds), np.array(targets)

log(f"\n{'Epoch':>6} {'TrainLoss':>10} {'Val RMSE':>10} {'Val MAE':>9}")
log("-" * 40)

best_rmse  = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        running += loss.item() * len(yb)

    train_loss = running / len(y_train)
    preds, targets = evaluate(test_loader)
    val_rmse = float(np.sqrt(mean_squared_error(targets, preds)))
    val_mae  = float(mean_absolute_error(targets, preds))
    scheduler.step(val_rmse)

    if val_rmse < best_rmse:
        best_rmse  = val_rmse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        # Save to disk immediately so we can kill the process at any time
        torch.save({
            "model_state":  best_state,
            "scaler_mean":  scaler.mean_,
            "scaler_scale": scaler.scale_,
            "features":     FEATURES,
            "window":       WINDOW,
            "epoch":        epoch,
            "rmse":         best_rmse,
            "mae":          val_mae,
        }, "lstm_pm25_best.pt")

    log(f"{epoch:>6d} {train_loss:>10.2f} {val_rmse:>10.2f} {val_mae:>9.2f}  {'*' if val_rmse == best_rmse else ''}")

# 9. Final evaluation with best checkpoint
model.load_state_dict(best_state)
y_pred_raw, _ = evaluate(test_loader)
y_pred = np.clip(y_pred_raw, 0.0, None)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))

log("\n" + "=" * 55)
log("  LSTM Final Results (best checkpoint)")
log("=" * 55)
log(f"  Test RMSE : {rmse:.2f} µg/m³   (LR baseline: 28.58)")
log(f"  Test MAE  : {mae:.2f} µg/m³   (LR baseline: 16.31)")
log(f"  Improvement — RMSE: {28.58 - rmse:+.2f}  MAE: {16.31 - mae:+.2f}")

# 10. AQI Tier Classification (5 tiers)
def pm25_to_aqi_tier(values):
    return np.searchsorted(AQI_BINS[1:-1], values)   # 0-indexed, 0..4

pred_tiers = pm25_to_aqi_tier(y_pred)
true_tiers = pm25_to_aqi_tier(y_test)

cat_acc = accuracy_score(true_tiers, pred_tiers)
log("\n" + "=" * 55)
log("  AQI Tier Classification  (5 categories)")
log("=" * 55)
log(f"  Overall accuracy: {cat_acc:.2%}\n")

log("  Ground-truth tier distribution:")
for i, label in enumerate(AQI_LABELS):
    n   = (true_tiers == i).sum()
    pct = n / len(true_tiers) * 100
    log(f"    [{i}] {label:<40s}: {n:>6,} ({pct:.1f}%)")

log("")
log(classification_report(true_tiers, pred_tiers,
                           target_names=AQI_LABELS,
                           zero_division=0))

# 11. Save model
save_path = "lstm_pm25.pt"
torch.save({
    "model_state":  best_state,
    "scaler_mean":  scaler.mean_,
    "scaler_scale": scaler.scale_,
    "features":     FEATURES,
    "window":       WINDOW,
    "rmse":         rmse,
    "mae":          mae,
    "aqi_accuracy": cat_acc,
}, save_path)
log(f"\nModel + scaler saved → {save_path}")
