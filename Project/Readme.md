#  주제: 기계학습 기반 주가 방향 예측 및 거래비용을 고려한 자동매매 백테스트 프로젝트
 
#### 기계학습과응용 / 인형진 (2021131028)

---

## 1. 모티베이션 (프로젝트를 하게 된 동기)

저는 이전 학기에 ‘수학과프로그래밍’ 수업에서 주가 예측 기반 자동매매 시뮬레이션 프로젝트를 진행하며, 예측 모델을 만들고 매매 규칙을 코드로 구현해보는 경험을 했습니다. 그러나 해당 프로젝트는 시뮬레이션 데이터와 단순한 점화식 기반 예측 구조에 머물러 있어, 실제 금융 데이터에서의 성능이나 다양한 예측 모델 간 비교까지는 다루지 못했습니다.

이번 학기 ‘기계학습과응용’ 수업에서는 Logistic Regression, Decision Tree, Random Forest, SVM, 그리고 PyTorch 기반의 신경망 모델 등 다양한 기계학습 기법을 학습하였습니다. 이에 따라 이전 프로젝트를 발전시켜, 실제 주가 데이터를 기반으로 시계열 feature를 설계하고, 여러 기계학습 모델을 동일한 조건에서 비교하며, 예측 성능뿐 아니라 거래비용(commission/slippage)을 포함한 투자 성과까지 함께 분석하는 프로젝트를 진행하게 되었습니다.

---

## 2. 이론적 배경

* **이진 분류 (Binary Classification)**: 다음 시점의 로그수익률이 양수인지 여부를 예측하여, 상승(1) / 하락(0) 형태의 분류 문제로 모델을 학습합니다.
* **시계열 Feature Engineering**: 로그수익률(ret), 과거 수익률 지연(ret_lag), 이동평균 대비 비율(ma_ratio), 변동성(vol), 모멘텀(mom) 등의 지표로 입력 변수를 구성합니다.
* **다중 기계학습 모델 비교**: Logistic Regression, Decision Tree, Random Forest, SVM, MLP(Pytorch)를 동일한 데이터/feature로 학습하여 예측 성능과 투자 성과를 비교합니다.
* **거래비용을 고려한 백테스트**: 매수·매도 시 수수료(commission) 및 슬리피지(slippage)를 반영하여 자동매매 성과를 평가합니다.
* **시계열 분할(Time-based split)**: 미래 정보 누수(leakage)를 방지하기 위해 시간 순서대로 Train/Test를 분할합니다.

---

## 3. 코드 작성방법 및 설명

### 핵심 구성 요소: 총 5개의 구성요소(함수/클래스) 기반 구조

---

### 1. `load_price_yf`

> **역할**: yfinance를 통해 실제 주가 데이터를 다운로드하고, `date`, `price` 형태의 DataFrame으로 변환합니다.

* `yf.download(..., auto_adjust=True)`: 배당/분할 영향을 반영한 가격을 사용합니다.
* `MultiIndex` 컬럼이 생기는 경우를 처리하여 `Close` 단일 컬럼을 안정적으로 사용합니다.
* 최종적으로 `date`, `price` 컬럼만 갖는 DataFrame을 반환합니다.

```python
def load_price_yf(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

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
2. FeatureConfig / build_features
역할: 주가 시계열로부터 기계학습 입력 변수(feature)와 타깃 변수(target)를 생성합니다.

log_price, ret: 로그가격과 로그수익률을 계산합니다.

ret_lag_k: 과거 수익률 지연 변수를 생성합니다. (기본 10개)

ma_ratio_w: 이동평균 대비 현재가격 비율을 생성합니다. (기본 5, 20)

vol_w: 수익률의 rolling 표준편차(변동성)를 생성합니다. (기본 5, 20)

mom_w: 모멘텀(과거 대비 가격 변화율)을 생성합니다. (기본 5, 20)

y: 다음 시점 수익률이 양수면 1, 아니면 0으로 라벨링합니다.

python
코드 복사
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
3. TorchMLP / TorchMLPModel
역할: PyTorch 기반의 MLP 이진 분류 모델을 구현하고, 상승 확률 P(up)을 예측합니다.

TorchMLP: 64-64 hidden layer를 갖는 간단한 신경망 구조입니다.

TorchMLPModel: 입력 표준화(StandardScaler) 후 학습하며, BCEWithLogitsLoss로 최적화합니다.

predict_proba: sigmoid를 적용해 확률 형태의 예측값을 반환합니다.

python
코드 복사
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
4. BacktestConfig / backtest
역할: 예측 확률 기반 Long/Flat 자동매매를 수행하고, 거래비용을 반영한 성과를 계산합니다.

매수 조건: P(up) >= 0.55이면 매수(Long)

매도 조건: P(up) <= 0.45이면 매도(현금화, Flat)

거래비용 반영:

매수 시: commission + slippage만큼 비용 차감 후 매수

매도 시: 매도금액에서 commission + slippage 차감 후 현금화

성과 지표:

CumRet(누적 수익률), MDD(최대 낙폭), Sharpe Ratio

Equity Curve(자산曲線), Drawdown Curve(낙폭曲線)

python
코드 복사
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
5. run
역할: 전체 파이프라인을 실행하고, 모델 비교 결과를 표(DataFrame) 및 그래프로 출력합니다.

AAPL 데이터를 다운로드하고 feature를 생성합니다.

시간 순서대로 Train/Test(70/30) 분할을 수행합니다.

모델별로 예측 확률을 얻어(상승 확률 P(up)), 예측 성능 지표(ACC, F1, AUC)와 백테스트 성과(CumRet, MDD, Sharpe)를 계산합니다.

결과를 DataFrame 형태로 출력하고, Equity Curve 및 Drawdown Curve를 시각화합니다.

python
코드 복사
run()
4. 프로젝트의 한계
기술적 지표 중심의 feature만 사용하여 거시경제 변수나 뉴스 정보는 반영하지 못함

단일 종목(AAPL) 중심의 실험으로 일반화에는 한계가 있음

단순 Long/Flat 전략으로 포지션 사이징 및 고급 리스크 관리는 포함하지 않음

거래비용을 고정값으로 가정하여 실제 시장의 복잡한 체결 구조는 완전히 반영하지 못함

5. 실행 방법 (Colab 기준)
python
코드 복사
# 1) 라이브러리 설치
!pip -q install yfinance

# 2) 코드 실행
run()
6. 결론 및 개선점
본 프로젝트에서는 실제 주가 데이터를 대상으로 시계열 기반 feature를 설계하고, 여러 기계학습 모델을 동일한 조건에서 비교했습니다. 또한 예측 정확도뿐 아니라 거래비용을 포함한 백테스트 성과를 함께 분석함으로써, 예측 성능과 실제 투자 성과가 반드시 일치하지는 않는다는 점을 확인할 수 있었습니다.

향후에는 다음과 같은 방향으로 개선할 수 있을 것입니다:

다양한 종목(또는 포트폴리오)으로 확장하여 일반화 성능 확인

확률 임계치(p_buy, p_sell) 최적화 및 거래 빈도 제어

포지션 사이징 및 리스크 관리 규칙 추가

LSTM/Transformer 등 시계열 딥러닝 모델 적용 및 비교

거래비용을 시장 상황에 따라 변하는 형태로 모델링(유동성 기반 슬리피지 등)

사용한 라이브러리
python
코드 복사
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
