# Forecasting Urban Air Quality — ECE 542 Final Project

LSTM-based PM2.5 forecasting on the Beijing Multi-Site Air Quality Dataset.  
Adam Davis · Austin Jenkins · Sebastian King

---

## Setup

**Dependencies**
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

**Data**  
Download the [Beijing Multi-Site Air Quality Dataset](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data) and place the extracted folder at:
```
beijing+multi+site+air+quality+data/PRSA_Data_20130301-20170228/
```

---

## Running the Models

### Linear Regression Baseline
```bash
python baseline_linear_regression.py
```
Trains on all 12 stations, evaluates on the held-out test set (Apr 2016 – Feb 2017), and prints RMSE, MAE, and AQI tier accuracy.

---

### Pooled LSTM (all 12 stations)
```bash
python lstm_forecaster_v1.py
```
Trains a 2-layer LSTM (128 hidden units, 128→64→1 FC head) on all stations combined.  
Saves best checkpoint by test RMSE to `lstm_pm25_best.pt`. Full AQI tier evaluation is printed at the end of the run.

---

### Per-Station LSTM (Dingling)
```bash
python lstm_per_station.py Dingling
```
Trains a 2-layer LSTM with temporal self-attention (128 hidden units, 128→32→1 FC head) on Dingling station data only.  
Saves best checkpoint to `lstm_Dingling_best.pt`.

Any of the 12 station names can be substituted as the argument.

---

## Saved Checkpoints

Pre-trained checkpoints are included. Each `.pt` file contains the model weights, StandardScaler parameters, and all metadata needed for inference — no retraining required.

| File                    | Model                       | Test RMSE   | Test MAE   |
| `lstm_pm25_best.pt`     | Pooled LSTM (12 stations)   | 17.47 μg/m³ | 9.35 μg/m³ |
| `lstm_Dingling_best.pt` | Per-station LSTM (Dingling) | 16.54 μg/m³ | 8.69 μg/m³ |

To load a checkpoint for inference:
```python
import torch
ck = torch.load('lstm_Dingling_best.pt', map_location='cpu', weights_only=False)
# ck['model_state']  — model weights
# ck['scaler_mean'], ck['scaler_scale']  — StandardScaler parameters
# ck['y_mean'], ck['y_std']  — target denormalization (per-station only)
# ck['features']  — ordered list of input features
# ck['window']    — input sequence length (24)
```

---

## Train / Test Split

All models use a strict temporal split with no shuffling:
- **Train:** March 2013 – March 2016
- **Test:** April 2016 – February 2017
