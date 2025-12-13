# 📈 기계학습 기반 주가 예측 및 자동매매 시뮬레이션 프로젝트
#### 기계학습과 응용 / 인형진 (2021131028)

---

## 1. 모티베이션 (프로젝트를 하게 된 동기)

지난 학기 ‘수학과 프로그래밍’ 수업에서 점화식을 활용한 주가 예측 시뮬레이터를 개발한 경험이 있습니다. 당시 프로젝트는 수학적 모델을 코드로 구현하는 데에는 성공했으나, **랜덤하게 생성된 가상 데이터(Random Walk)**를 사용했다는 점과 단순한 **선형 점화식**에 의존했다는 한계가 있었습니다.

이번 학기 **<기계학습과 응용>** 수업을 통해, 실제 금융 시장의 데이터는 훨씬 복잡하고 비선형적인 패턴(Non-linear Pattern)을 가지고 있음을 배웠습니다. 이에 이전 프로젝트를 발전시켜 다음과 같은 목표로 본 프로젝트를 새롭게 기획하였습니다.

1.  **Real World Data:** 가상 데이터가 아닌 **Yahoo Finance API**를 연동하여 실제 주식(Apple Inc.) 데이터를 분석합니다.
2.  **Deep Learning:** 단순 회귀를 넘어 **PyTorch 기반의 MLP(Multi-Layer Perceptron)** 모델을 직접 설계하고 적용합니다.
3.  **Risk Management:** **조기 종료(Early Stopping)**와 **거래 비용(Commission)** 개념을 도입하여 현실적이고 견고한 자동매매 시스템을 구축합니다.

---

## 2. 이론적 배경

* **이진 분류 (Binary Classification):** 주가를 정확히 맞추는 회귀(Regression) 대신, 내일 주가가 오를지 내릴지를 예측하는 분류 문제로 접근하여 예측의 안정성을 높였습니다.
* **피처 엔지니어링 (Feature Engineering):** 단순히 '가격'만 보는 것이 아니라, 금융 도메인 지식을 활용해 **이동평균 괴리율(MA Ratio)**, **변동성(Volatility)**, **모멘텀(Momentum)**, **지연 수익률(Lagged Return)** 등을 입력 변수로 가공하여 사용했습니다.
* **MLP와 조기 종료 (Early Stopping):** 딥러닝 모델이 학습 데이터만 외우는 **과적합(Overfitting)**을 방지하기 위해, 검증(Validation) 오차가 줄어들지 않으면 학습을 중단하는 기법을 적용했습니다.

---

## 3. 코드 작성방법 및 설명

전체 코드는 객체 지향 프로그래밍(OOP) 구조로 설계되었으며, 데이터 수집부터 백테스팅까지 파이프라인이 구축되어 있습니다.

### 핵심 구성 요소: 데이터 처리 및 모델링

### 1. `FeatureConfig` & `build_features`

> **역할**: 원본 주가 데이터를 머신러닝 모델이 학습할 수 있는 형태의 풍부한 파생 변수(Feature)로 변환합니다.

* `FeatureConfig`: 이동평균 기간(5일, 20일), 모멘텀 윈도우 등 하이퍼파라미터를 관리하는 설정 클래스입니다.
* `build_features(df, cfg)`: 로그 수익률(`log_price`), 과거 수익률(`lag`), 이동평균 대비 비율(`ma_ratio`), 변동성(`vol`) 등을 벡터화 연산으로 고속 처리하여 생성합니다. 타겟 변수(`y`)는 다음 날 수익률이 0보다 크면 1, 아니면 0으로 설정합니다.

```python
def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    
    # ... (로그 가격 및 수익률 계산) ...

    # Moving Average Ratio 생성
    for w in cfg.ma_windows:
        ma = out["price"].rolling(w).mean()
        out[f"ma_ratio_{w}"] = out["price"] / ma - 1.0
    
    # Target: Next Day Return > 0 (내일 오르면 1, 아니면 0)
    out["future_ret"] = out["ret"].shift(-1)
    out["y"] = (out["future_ret"] > 0).astype(int)
    return out.dropna().reset_index(drop=True)
```

### 2. `TorchMLPModel` (Deep Learning Core)

> **역할**: PyTorch를 활용해 심층 신경망을 구축하고, 과적합을 방지하며 학습을 수행합니다.

* **__init__**: 입력 차원에 맞춰 3층 신경망(Input → 64 → 64 → 1)을 설계합니다. `ReLU` 활성화 함수와 `Dropout(0.1)`을 적용해 일반화 성능을 높였습니다.
* **fit(X, y)**:
    * 데이터를 학습용과 검증용(20%)으로 분리(`train_test_split`)합니다.
    * **Early Stopping 로직**: 매 Epoch마다 검증 손실(Val Loss)을 확인하여, 20회 이상 성능이 개선되지 않으면 학습을 조기 종료하고 가장 성능이 좋았던 시점의 가중치로 복구합니다.

```python
class TorchMLPModel:
    # ... (초기화 코드 생략) ...
    def fit(self, X, y):
        # ... (데이터 분리 및 스케일링) ...
        for epoch in range(self.epochs):
            # ... (학습 및 검증 진행) ...
            
            # 조기 종료 (Early Stopping) 체크
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    print(f"*** Early Stopping at Epoch {epoch+1} ***")
                    break
        # 최적 가중치 복구
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
```

### 3. `BacktestConfig` & `backtest`

> **역할**: 예측된 확률을 기반으로 실제 매매를 시뮬레이션하고 성과를 측정합니다.

* **Threshold Strategy**: 모델이 예측한 상승 확률이 **60%(`p_buy`)** 이상일 때만 확실하다고 판단해 매수하고, **40%(`p_sell`)** 이하일 때만 매도합니다.
* **Transaction Costs**: 매매 시마다 0.05%의 수수료와 슬리피지를 차감하여 현실성을 높였습니다.

```python
def backtest(prices, probs, cfg: BacktestConfig):
    # ... (자산 초기화) ...
    for price, p in zip(prices, probs):
        # Buy Signal: 확률이 p_buy(0.6) 이상이고 주식이 없을 때
        if shares == 0 and p >= cfg.p_buy:
            cost = cash * (cfg.commission + cfg.slippage)
            shares = (cash - cost) / price
            cash = 0
            
        # Sell Signal: 확률이 p_sell(0.4) 이하이고 주식을 보유 중일 때
        elif shares > 0 and p <= cfg.p_sell:
            cash = shares * price * (1 - cfg.commission - cfg.slippage)
            shares = 0
            
    # ... (수익률, MDD, Sharpe Ratio 계산 및 반환) ...
```

---

## 4. 실행 예시 및 결과 분석

**대상 종목:** Apple (AAPL)  
**테스트 기간:** 2022년 12월 ~ 2024년 12월

### 4.1. 모델 성능 비교표

| Model | Accuracy | CumRet (수익률) | MDD (최대낙폭) | Sharpe | Trades (매매횟수) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic** | **0.5421** | **0.8652 (86.5%)** | **-0.1661** | **1.5100** | **1** |
| Benchmark | - | 0.7184 (71.8%) | -0.22XX | - | - |
| RF | 0.5383 | 0.3226 (32.2%) | -0.1661 | 0.8363 | 1 |
| SVM | 0.5517 | 0.0000 (0.0%) | 0.0000 | 0.0000 | 0 |
| MLP | 0.5421 | 0.0000 (0.0%) | 0.0000 | 0.0000 | 0 |

### 4.2. 자산 가치 변화 (Equity Curve)
![Equity Curve](image_08d3f7.png)
*(실제 실행 결과로 도출된 자산 그래프: Logistic 모델이 Benchmark를 상회하는 성과를 보임)*

### 4.3. 변수 중요도 (Feature Importance)
![Feature Importance](image_08d3c2.png)
*(Random Forest 분석 결과: 최근 1~2일보다 8~9일 전의 수익률(`ret_lag_9`)이 예측에 중요한 영향을 미침)*

### 4.4. 결과 분석

* **단순함의 승리 (Logistic Regression):** 실험 기간 동안 시장은 전반적인 상승장이었습니다. 선형 모델인 로지스틱 회귀는 이러한 '상승 추세(Trend)'를 가장 잘 포착하여, 초기에 매수한 뒤 계속 보유하는 전략으로 가장 높은 수익(86.5%)을 달성했습니다.
* **딥러닝의 신중함 (MLP):** MLP 모델은 Early Stopping을 통해 과적합을 방지하도록 학습되었습니다. 그 결과, 시장의 노이즈 속에서 60% 이상의 확신을 주는 패턴을 찾지 못했고(예측 확률이 54% 수준에 머무름), **"확실하지 않으면 투자하지 않는다"**는 원칙에 따라 매매를 하지 않아 자산을 보존했습니다.

---

## 5. 결론 및 개선점

이번 프로젝트를 통해 **"복잡한 모델이 항상 더 나은 수익을 보장하지 않는다"**는 금융 데이터 분석의 중요한 교훈을 얻었습니다. 딥러닝 모델은 강력하지만, 노이즈가 심한 주가 데이터에서는 과적합을 피하기 위해 매우 보수적으로 작동할 수 있음을 확인했습니다. 또한, 단순한 예측 정확도보다 **임계값(Threshold) 설정**과 **수수료 관리**가 실제 수익률에 더 큰 영향을 미친다는 것을 배웠습니다.

**향후 개선 방향:**
* **Threshold 최적화:** 현재 설정된 진입 장벽(0.6)이 너무 높았습니다. Grid Search를 통해 모델별 최적의 매매 임계값을 찾는다면 딥러닝 모델의 잠재력을 더 끌어낼 수 있을 것입니다.
* **포트폴리오 확장:** 단일 종목(AAPL)이 아닌 S&P 500 전 종목을 대상으로 학습한다면, 개별 종목의 노이즈를 줄이고 시장 전체의 일반적인 패턴을 학습할 수 있을 것입니다.
* **시계열 특화 모델:** MLP를 넘어 LSTM이나 Transformer 구조를 도입하여 더 긴 기간의 시간적 의존성을 분석해보고 싶습니다.

---

## 6. 사용한 라이브러리

본 프로젝트는 다음의 라이브러리들을 사용하여 작성되었습니다.

```python
import yfinance as yf          # 주가 데이터 수집
import pandas as pd            # 데이터 전처리
import numpy as np             # 수치 연산
import matplotlib.pyplot as plt # 시각화
import torch                   # 딥러닝 프레임워크 (PyTorch)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
```
