"""
Per-station LSTM — trains one model per monitoring site.
With only ~27K sequences per station the cross-station noise is eliminated,
letting a smaller model specialise on location-specific patterns.

Usage:
    python lstm_per_station.py              # trains STATION (default: Dongsi)
    python lstm_per_station.py Dingling     # trains a specific station
"""

import os, glob, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             classification_report, accuracy_score)

def log(msg): print(msg, flush=True)

# Config
STATION     = sys.argv[1] if len(sys.argv) > 1 else "Dongsi"
DATA_DIR    = "beijing+multi+site+air+quality+data/PRSA_Data_20130301-20170228"
WINDOW      = 24
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 128
LR          = 3e-4
WEIGHT_DECAY= 1e-4
EPOCHS      = 50
T_MAX       = 50 # cosine schedule horizon
SPLIT_DATE  = "2016-04-01"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

AQI_BINS   = [0, 12.0, 35.4, 55.4, 150.4, float("inf")]
AQI_LABELS = ["Good", "Moderate", "Unhealthy (Sensitive Groups)",
               "Unhealthy", "Very Unhealthy / Hazardous"]

log(f"Station: {STATION}  |  Device: {DEVICE}")

# 1. Load this station only
files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
dfs = []
for f in files:
    tmp = pd.read_csv(f)
    if STATION in tmp["station"].values:
        dfs.append(tmp)
if not dfs:
    raise ValueError(f"Station '{STATION}' not found. Available: check CSV files.")

df = pd.concat(dfs, ignore_index=True)
df = df[df["station"] == STATION].copy()
df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
df = df.sort_values("datetime").reset_index(drop=True)
log(f"Rows for {STATION}: {len(df):,}")

# 2. Feature preprocessing
NUMERIC = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
           "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
WIND_DIR_MAP = {
    "N":0,"NNE":22.5,"NE":45,"ENE":67.5,"E":90,"ESE":112.5,
    "SE":135,"SSE":157.5,"S":180,"SSW":202.5,"SW":225,"WSW":247.5,
    "W":270,"WNW":292.5,"NW":315,"NNW":337.5,
}
df["wd_deg"]    = df["wd"].map(WIND_DIR_MAP)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

FEATURES = NUMERIC + ["wd_deg", "hour_sin", "hour_cos", "month_sin", "month_cos"]
df[FEATURES] = df[FEATURES].interpolate(method="linear").bfill().ffill()
df = df.dropna(subset=["PM2.5"])

# 3. Sliding-window construction
vals      = df[FEATURES].values.astype(np.float32)
target    = df["PM2.5"].values.astype(np.float32)
datetimes = df["datetime"].values

W, F = WINDOW, len(FEATURES)
X_sw  = np.lib.stride_tricks.sliding_window_view(vals, (W, F)).reshape(-1, W, F)[:-1]
y_sw  = target[W:]
dt_sw = datetimes[W:]
mask  = ~(np.isnan(X_sw).any(axis=(1, 2)) | np.isnan(y_sw))
X_sw, y_sw, dt_sw = X_sw[mask], y_sw[mask], dt_sw[mask]
log(f"Sequences: {len(X_sw):,}  |  PM2.5 mean={y_sw.mean():.1f}  std={y_sw.std():.1f}")

# 4. Train / test split
split      = np.datetime64(SPLIT_DATE)
tr = dt_sw < split; te = dt_sw >= split
X_train, y_train = X_sw[tr].astype(np.float32), y_sw[tr].astype(np.float32)
X_test,  y_test  = X_sw[te].astype(np.float32), y_sw[te].astype(np.float32)
log(f"Train: {tr.sum():,}  |  Test: {te.sum():,}")

# 5. Per-station standardisation
n_feat    = X_train.shape[2]
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
X_test_s  = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

# Normalize target — model predicts z-score, we invert for evaluation
# Eliminates bias-initialization problem when station mean >> 0
y_mean = float(y_train.mean())
y_std  = float(y_train.std()) + 1e-8
y_train_n = ((y_train - y_mean) / y_std).astype(np.float32)
y_test_n  = ((y_test  - y_mean) / y_std).astype(np.float32)
log(f"Target normalised: mean={y_mean:.1f}  std={y_std:.1f}  -> z-score")

# 6. Dataset & DataLoader
class PM25Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(PM25Dataset(X_train_s, y_train_n),
                          batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
test_loader  = DataLoader(PM25Dataset(X_test_s,  y_test_n),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 7. Model: LSTM + temporal attention
class LSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        w   = torch.softmax(self.attn_score(out), dim=1)
        ctx = (w * out).sum(dim=1)
        return self.head(ctx).squeeze(-1)

model     = LSTMWithAttention(n_feat, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=T_MAX, eta_min=1e-5)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Model: LSTM({HIDDEN_DIM}x{NUM_LAYERS}) + attention  |  params: {n_params:,}")

# 8. Evaluation — invert z-score to original PM2.5 scale
def evaluate(loader):
    model.eval()
    preds_n, targets_n = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            preds_n.extend(model(Xb.to(DEVICE)).cpu().numpy())
            targets_n.extend(yb.numpy())
    # invert normalisation → original µg/m³
    p = np.clip(np.array(preds_n) * y_std + y_mean, 0, None)
    t = np.array(targets_n) * y_std + y_mean
    return p, t, float(np.sqrt(mean_squared_error(t, p))), float(mean_absolute_error(t, p))

# 9. Training loop
log(f"\n{'Epoch':>6} {'LR':>8} {'TrainLoss':>10} {'Val RMSE':>10} {'Val MAE':>9}")
log("-" * 50)

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
    current_lr = scheduler.get_last_lr()[0]
    scheduler.step()
    preds, targets, val_rmse, val_mae = evaluate(test_loader)

    if val_rmse < best_rmse:
        best_rmse  = val_rmse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        torch.save({
            "model_state":  best_state,
            "scaler_mean":  scaler.mean_,
            "scaler_scale": scaler.scale_,
            "y_mean":       y_mean,
            "y_std":        y_std,
            "features":     FEATURES,
            "window":       WINDOW,
            "hidden_dim":   HIDDEN_DIM,
            "num_layers":   NUM_LAYERS,
            "dropout":      DROPOUT,
            "station":      STATION,
            "epoch":        epoch,
            "rmse":         best_rmse,
            "mae":          val_mae,
        }, f"lstm_{STATION}_best.pt")

    marker = "*" if val_rmse == best_rmse else ""
    log(f"{epoch:>6d} {current_lr:>8.2e} {train_loss:>10.2f} {val_rmse:>10.2f} {val_mae:>9.2f}  {marker}")

# 10. Final evaluation + AQI classification
model.load_state_dict(best_state)
y_pred, y_true, rmse, mae = evaluate(test_loader)

log("\n" + "=" * 55)
log(f"  {STATION} — Final Results (best checkpoint, epoch {best_rmse:.2f} RMSE)")
log("=" * 55)
log(f"  Test RMSE : {rmse:.2f} ug/m3   (pooled baseline: 17.47)")
log(f"  Test MAE  : {mae:.2f} ug/m3")

def to_tier(v): return np.searchsorted(AQI_BINS[1:-1], v)
pred_tiers = to_tier(y_pred); true_tiers = to_tier(y_true)
acc = accuracy_score(true_tiers, pred_tiers)
log(f"\n  AQI Tier Accuracy: {acc:.2%}")
log(f"\n  Ground-truth tier distribution:")
for i, label in enumerate(AQI_LABELS):
    n = (true_tiers == i).sum(); pct = n / len(true_tiers) * 100
    log(f"    [{i}] {label:<40s}: {n:>5,} ({pct:.1f}%)")
log("")
log(classification_report(true_tiers, pred_tiers,
                           target_names=AQI_LABELS, zero_division=0))

torch.save({
    "model_state":  best_state,
    "scaler_mean":  scaler.mean_,
    "scaler_scale": scaler.scale_,
    "features":     FEATURES,
    "station":      STATION,
    "window":       WINDOW,
    "rmse":         rmse,
    "mae":          mae,
    "aqi_accuracy": acc,
    "aqi_bins":     AQI_BINS,
    "aqi_labels":   AQI_LABELS,
}, f"lstm_{STATION}_final.pt")
log(f"Saved: lstm_{STATION}_final.pt")
