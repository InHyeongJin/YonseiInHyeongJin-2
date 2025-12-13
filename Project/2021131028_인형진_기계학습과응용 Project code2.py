!pip -q install yfinance

# ============================================================
# ML Trading Project - FINAL COMPLETE VERSION (w/ Early Stopping & Feature Importance)
# ============================================================

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple, Dict
from copy import deepcopy 

# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split 

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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


# ============================================================
# Data Loader (yfinance 안정 처리)
# ============================================================

def load_price_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    print(f"Downloading data for {ticker}...")
    data = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

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

    # Lag features
    for k in range(1, cfg.lags + 1):
        out[f"ret_lag_{k}"] = out["ret"].shift(k)

    # Moving Average Ratio
    for w in cfg.ma_windows:
        ma = out["price"].rolling(w).mean()
        out[f"ma_ratio_{w}"] = out["price"] / ma - 1.0

    # Volatility
    for w in cfg.vol_windows:
        out[f"vol_{w}"] = out["ret"].rolling(w).std()

    # Momentum
    for w in cfg.mom_windows:
        out[f"mom_{w}"] = out["price"] / out["price"].shift(w) - 1.0

    # Target: Next Day Return > 0
    out["future_ret"] = out["ret"].shift(-1)
    out["y"] = (out["future_ret"] > 0).astype(int)

    return out.dropna().reset_index(drop=True)


# ============================================================
# Torch MLP (Improved with Early Stopping)
# ============================================================

class TorchMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class TorchMLPModel:
    def __init__(self, in_dim):
        self.scaler = StandardScaler()
        self.model = TorchMLP(in_dim)
        self.epochs = 500 # 조기 종료를 고려해 넉넉하게 설정
        self.patience = 20 # 20번의 에포크 동안 개선 없으면 종료

    def fit(self, X, y):
        set_seed(42)
        # 훈련 데이터에서 검증 데이터(20%) 분리
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 스케일링
        X_tr_s = self.scaler.fit_transform(X_tr).astype(np.float32)
        X_val_s = self.scaler.transform(X_val).astype(np.float32)
        y_tr = y_tr.astype(np.float32)
        y_val = y_val.astype(np.float32)

        ds = TensorDataset(torch.tensor(X_tr_s), torch.tensor(y_tr))
        dl = DataLoader(ds, batch_size=64, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        best_weights = deepcopy(self.model.state_dict())
        counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            # 훈련 (Training)
            for xb, yb in dl:
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
            
            # 검증 (Validation)
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(torch.tensor(X_val_s))
                val_loss = loss_fn(val_pred, torch.tensor(y_val)).item()
            self.model.train()

            # 조기 종료 로직 (Early Stopping Logic)
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    print(f"*** Early Stopping at Epoch {epoch+1} (Best Val Loss: {best_loss:.4f}) ***")
                    break
            
            if (epoch+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")

        # 최적의 가중치 복구
        if best_weights is not None:
            self.model.load_state_dict(best_weights)

    def predict_proba(self, X):
        self.model.eval()
        Xs = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            return torch.sigmoid(self.model(torch.tensor(Xs))).numpy()


# ============================================================
# Backtest (With Costs & Buy-Hold)
# ============================================================

@dataclass
class BacktestConfig:
    p_buy: float = 0.60
    p_sell: float = 0.40
    commission: float = 0.0005
    slippage: float = 0.0005
    initial_cash: float = 10000.0

def backtest(prices, probs, cfg: BacktestConfig):
    cash = cfg.initial_cash
    shares = 0.0
    equity = []
    trade_count = 0

    for price, p in zip(prices, probs):
        # Buy Signal
        if shares == 0 and p >= cfg.p_buy:
            cost = cash * (cfg.commission + cfg.slippage)
            shares = (cash - cost) / price
            cash = 0
            trade_count += 1
        
        # Sell Signal
        elif shares > 0 and p <= cfg.p_sell:
            cash = shares * price * (1 - cfg.commission - cfg.slippage)
            shares = 0
            trade_count += 1
            
        equity.append(cash + shares * price)

    equity = np.array(equity)
    drawdown = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity)
    ret = np.diff(equity) / equity[:-1]

    sharpe = np.mean(ret) / np.std(ret) * np.sqrt(252) if np.std(ret) > 0 else 0.0

    return {
        "CumRet": equity[-1] / equity[0] - 1,
        "MDD": drawdown.min(),
        "Sharpe": sharpe,
        "Equity": equity,
        "Drawdown": drawdown,
        "Trades": trade_count
    }


# ============================================================
# Run Experiment (Feature Importance Plot 추가)
# ============================================================

def run():
    set_seed(42)

    ticker = "AAPL"
    start = "2018-01-01"
    end = "2025-01-01"

    bt_cfg = BacktestConfig()

    # 1. Load & Feature Engineering
    try:
        df = load_price_yf(ticker, start, end)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    df_feat = build_features(df, FeatureConfig())

    feature_cols = [c for c in df_feat.columns if c.startswith(("ret_lag_", "ma_ratio_", "vol_", "mom_"))]

    X = df_feat[feature_cols].values
    y = df_feat["y"].values
    prices = df_feat["price"].values
    dates = df_feat["date"]

    split = int(0.7 * len(X))

    print("\n=== Train/Test Period ===")
    print(f"Train: {dates.iloc[0].date()} ~ {dates.iloc[split-1].date()}")
    print(f"Test : {dates.iloc[split].date()} ~ {dates.iloc[-1].date()}")

    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]
    prices_te = prices[split:]

    # 2. Benchmark (Buy & Hold) Calculation
    buy_hold_equity = prices_te / prices_te[0] * bt_cfg.initial_cash
    buy_hold_ret = (prices_te[-1] / prices_te[0]) - 1
    buy_hold_dd = (buy_hold_equity - np.maximum.accumulate(buy_hold_equity)) / np.maximum.accumulate(buy_hold_equity)

    models = {
        "Logistic": Pipeline([("scaler", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000))]),
        
        "RF": RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42),
        
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(probability=True))]),
        
        "MLP": TorchMLPModel(X.shape[1])
    }

    rows = []
    equity_curves = {}
    drawdowns = {}
    
    equity_curves["Buy&Hold"] = buy_hold_equity
    drawdowns["Buy&Hold"] = buy_hold_dd

    print("\nTraining models...")
    # RF 모델의 feature_importances_를 추출하기 위해 별도 저장
    rf_model = models["RF"]

    for name, model in models.items():
        print(f" -> {name}...")
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
            "CumRet": round(bt["CumRet"], 4),
            "MDD": round(bt["MDD"], 4),
            "Sharpe": round(bt["Sharpe"], 4),
            "Trades": bt["Trades"]
        })

        equity_curves[name] = bt["Equity"]
        drawdowns[name] = bt["Drawdown"]

    # 결과 테이블 출력
    result_df = pd.DataFrame(rows).sort_values("CumRet", ascending=False)
    
    print("\n=== Model Comparison Table ===")
    print(f"Benchmark (Buy&Hold) Return: {buy_hold_ret:.4f}")
    print(result_df)

    # --- 그래프 1: 자산 가치 변화 (Equity Curve) ---
    plt.figure(figsize=(12, 5))
    for name, eq in equity_curves.items():
        style = '--' if name == "Buy&Hold" else '-'
        alpha = 0.6 if name == "Buy&Hold" else 1.0
        width = 2.0 if name == "MLP" or name == "Buy&Hold" else 1.5
        plt.plot(eq, label=name, linestyle=style, linewidth=width, alpha=alpha)
        
    plt.title(f"Equity Curves (Test): Threshold {bt_cfg.p_buy}/{bt_cfg.p_sell}")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # --- 그래프 2: 낙폭 변화 (Drawdown Curve) ---
    plt.figure(figsize=(12, 4))
    for name, dd in drawdowns.items():
        style = '--' if name == "Buy&Hold" else '-'
        alpha = 0.6 if name == "Buy&Hold" else 1.0
        width = 2.0 if name == "MLP" or name == "Buy&Hold" else 1.5
        plt.plot(dd, label=name, linestyle=style, linewidth=width, alpha=alpha)

    plt.title("Drawdown Curves (Test)")
    plt.xlabel("Days")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # --- 그래프 3: Random Forest Feature Importance (추가됨) ---
    if hasattr(rf_model, 'feature_importances_'):
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = 10
        indices_top = indices[:top_n]

        plt.figure(figsize=(10, 6))
        plt.title(f"Feature Importances (Random Forest) - Top {top_n}")
        plt.bar(range(top_n), importances[indices_top], align="center")
        plt.xticks(range(top_n), [feature_cols[i] for i in indices_top], rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    else:
        print("\nRandom Forest 모델에서 Feature Importance를 찾을 수 없습니다.")

if __name__ == "__main__":
    run()
