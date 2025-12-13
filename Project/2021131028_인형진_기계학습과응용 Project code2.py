!pip -q install yfinance

# ============================================================
# ML Trading Project - FINAL STABLE VERSION (Colab)
# Table + Transaction Costs + Drawdown + Train/Test Dates
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Dict

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# Utils
# ============================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


# ============================================================
# Data Loader (yfinance 안정 처리)
# ============================================================

def load_price_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    # MultiIndex 제거
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    price = data["Close"].astype(float)

    df = pd.DataFrame({
        "date": price.index,
        "price": price.values
    }).dropna().reset_index(drop=True)

    return df


# ============================================================
# Feature Engineering
# ============================================================

@dataclass
class FeatureConfig:
    lags: int = 10
    ma_windows: Tuple[int, int] = (5, 20)
    vol_windows: Tuple[int, int] = (5, 20)
    mom_windows: Tuple[int, int] = (5, 20)

def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()

    out["log_price"] = np.log(out["price"])
    out["ret"] = out["log_price"].diff()

    for k in range(1, cfg.lags + 1):
        out[f"ret_lag_{k}"] = out["ret"].shift(k)

    for w in cfg.ma_windows:
        ma = out["price"].rolling(w).mean()
        out[f"ma_ratio_{w}"] = out["price"] / ma - 1.0

    for w in cfg.vol_windows:
        out[f"vol_{w}"] = out["ret"].rolling(w).std()

    for w in cfg.mom_windows:
        out[f"mom_{w}"] = out["price"] / out["price"].shift(w) - 1.0

    out["future_ret"] = out["ret"].shift(-1)
    out["y"] = (out["future_ret"] > 0).astype(int)

    return out.dropna().reset_index(drop=True)


# ============================================================
# Torch MLP
# ============================================================

class TorchMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchMLPModel:
    def __init__(self, in_dim):
        self.scaler = StandardScaler()
        self.model = TorchMLP(in_dim)
        self.epochs = 10

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X).astype(np.float32)
        y = y.astype(np.float32)

        ds = TensorDataset(torch.tensor(Xs), torch.tensor(y))
        dl = DataLoader(ds, batch_size=128, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(self.epochs):
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()

    def predict_proba(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(Xs))).numpy()


# ============================================================
# Backtest (with costs)
# ============================================================

@dataclass
class BacktestConfig:
    p_buy: float = 0.55
    p_sell: float = 0.45
    commission: float = 0.0005
    slippage: float = 0.0005
    initial_cash: float = 10000.0

def backtest(prices, probs, cfg: BacktestConfig):
    cash = cfg.initial_cash
    shares = 0.0
    equity = []

    for price, p in zip(prices, probs):
        if shares == 0 and p >= cfg.p_buy:
            cost = cash * (cfg.commission + cfg.slippage)
            shares = (cash - cost) / price
            cash = 0
        elif shares > 0 and p <= cfg.p_sell:
            cash = shares * price * (1 - cfg.commission - cfg.slippage)
            shares = 0
        equity.append(cash + shares * price)

    equity = np.array(equity)
    drawdown = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
    ret = np.diff(equity) / equity[:-1]

    sharpe = np.mean(ret) / np.std(ret) * np.sqrt(len(ret)) if np.std(ret) > 0 else np.nan

    return {
        "CumRet": equity[-1] / equity[0] - 1,
        "MDD": drawdown.min(),
        "Sharpe": sharpe,
        "Equity": equity,
        "Drawdown": drawdown
    }


# ============================================================
# Run Experiment
# ============================================================

def run():
    set_seed(42)

    ticker = "AAPL"
    start = "2018-01-01"
    end = "2025-01-01"

    bt_cfg = BacktestConfig()

    df = load_price_yf(ticker, start, end)
    df_feat = build_features(df, FeatureConfig())

    feature_cols = [c for c in df_feat.columns if c.startswith(("ret_lag_", "ma_ratio_", "vol_", "mom_"))]

    X = df_feat[feature_cols].values
    y = df_feat["y"].values
    prices = df_feat["price"].values
    dates = df_feat["date"]

    split = int(0.7 * len(X))

    print("=== Train/Test Period ===")
    print(f"Train: {dates.iloc[0].date()} ~ {dates.iloc[split-1].date()}")
    print(f"Test : {dates.iloc[split].date()} ~ {dates.iloc[-1].date()}")

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    prices_te = prices[split:]

    models = {
        "Logistic": Pipeline([("scaler", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000))]),
        "Tree": DecisionTreeClassifier(max_depth=5),
        "RF": RandomForestClassifier(n_estimators=200, max_depth=6),
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(probability=True))]),
        "MLP": TorchMLPModel(X.shape[1])
    }

    rows = []
    equity_curves = {}
    drawdowns = {}

    for name, model in models.items():
        if name == "MLP":
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)
        else:
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_te)[:, 1]

        y_hat = (p >= 0.5).astype(int)

        bt = backtest(prices_te, p, bt_cfg)

        rows.append({
            "Model": name,
            "ACC": round(accuracy_score(y_te, y_hat), 4),
            "F1": round(f1_score(y_te, y_hat), 4),
            "AUC": round(safe_auc(y_te, p), 4),
            "CumRet": round(bt["CumRet"], 4),
            "MDD": round(bt["MDD"], 4),
            "Sharpe": round(bt["Sharpe"], 4)
        })

        equity_curves[name] = bt["Equity"]
        drawdowns[name] = bt["Drawdown"]

    result_df = pd.DataFrame(rows).sort_values("CumRet", ascending=False)
    print("\n=== Model Comparison Table ===")
    print(result_df)

    # Equity plot
    plt.figure(figsize=(10, 5))
    for name, eq in equity_curves.items():
        plt.plot(eq, label=name)
    plt.title("Equity Curves (Test)")
    plt.legend()
    plt.grid()
    plt.show()

    # Drawdown plot
    plt.figure(figsize=(10, 4))
    for name, dd in drawdowns.items():
        plt.plot(dd, label=name)
    plt.title("Drawdown Curves (Test)")
    plt.legend()
    plt.grid()
    plt.show()


# ============================================================
# Execute
# ============================================================

run()
