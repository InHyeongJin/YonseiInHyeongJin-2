# 📈 딥러닝 기반 주가 예측 및 자동매매 시스템
### 기계학습과 응용 기말 프로젝트 / [본인 이름] ([본인 학번])

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red) ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)

---

## 📑 목차 (Table of Contents)
1. [프로젝트 개요 (Overview)](#1-프로젝트-개요-overview)
2. [이론적 배경 및 방법론 (Methodology)](#2-이론적-배경-및-방법론-methodology)
3. [시스템 설계 및 구현 (Implementation)](#3-시스템-설계-및-구현-implementation)
4. [실험 환경 및 결과 (Experiments & Results)](#4-실험-환경-및-결과-experiments--results)
5. [결과 분석 및 고찰 (Discussion)](#5-결과-분석-및-고찰-discussion)
6. [결론 (Conclusion)](#6-결론-conclusion)

---

## 1. 프로젝트 개요 (Overview)

### 1.1. 배경 및 동기 (Motivation)
지난 학기 ‘수학과 프로그래밍’ 수업에서 **점화식(Recurrence Relation)** 활용한 주가 예측 시뮬레이터를 개발한 경험이 있습니다. 당시 프로젝트는 수학적 모델을 코드로 구현하는 과정 자체에는 성공했으나, **랜덤하게 생성된 가상 데이터 (Random Walk)** 를 사용했다는 점과 단순한 **선형 모델**에만 의존했다는 근본적인 한계가 있었습니다. 이는 실제 금융 시장의 고차원적 복잡성과 불확실성을 반영하기에는 역부족이었습니다.

### 1.2. 프로젝트 목표 (Objectives)
이번 학기 **<기계학습과 응용>** 수업을 통해, 실제 데이터는 훨씬 비선형적인 패턴(Non-linear Pattern)을 내포하고 있음을 학습했습니다. 이에 본 프로젝트는 이전의 경험을 발전시켜 다음과 같은 심화된 목표를 설정하였습니다.

1.  **Real-World Integration:** 가상의 데이터가 아닌 **Yahoo Finance API**를 연동하여, 실제 시장(Apple Inc.)의 Historical Data를 수집하고 전처리하는 파이프라인을 구축합니다.
2.  **Advanced Modeling:** 단순 회귀 분석을 넘어, **PyTorch 기반의 MLP (Multi-Layer Perceptron)** 모델을 직접 설계하고 학습시켜 딥러닝의 효용성을 검증합니다.
3.  **Robust Backtesting:** 단순히 예측 정확도만 높이는 것이 아니라, **조기 종료 (Early Stopping)** 기법을 통해 과적합을 제어하고, **거래 비용 (Commission & Slippage)** 을 반영한 현실적인 자동매매 시뮬레이션을 수행합니다.

---

## 2. 이론적 배경 및 방법론 (Methodology)

### 2.1. 문제 정의 (Problem Definition)
본 프로젝트는 주식 시장의 예측 불가능성을 고려하여, 정확한 가격(Price)을 맞추는 회귀(Regression) 문제가 아닌, **익일 주가의 등락 방향(Up/Down)을 예측하는 이진 분류(Binary Classification)** 문제로 접근하였습니다.

### 2.2. 데이터 전처리 및 피처 엔지니어링 (Feature Engineering)
금융 시계열 데이터의 특성인 비정상성(Non-stationarity)을 극복하고 모델의 학습 효율을 높이기 위해 다음과 같은 파생 변수(Derived Features)를 생성하였습니다.

* **로그 수익률 (Log Returns):** 주가의 절대적인 크기에 영향을 받지 않도록 로그 차분(Log-Difference)을 사용하여 데이터의 분포를 정규화했습니다.
* **지연 수익률 (Lagged Returns):** 시계열 데이터의 자기상관성(Autocorrelation)을 반영하기 위해 과거 1일~10일 전의 수익률을 입력 변수로 사용했습니다.
* **이동평균 괴리율 (MA Ratio):** 5일, 20일 이동평균선과 현재 주가 간의 거리를 측정하여 추세(Trend) 정보를 반영했습니다.
* **변동성 (Volatility):** 최근 가격의 표준편차를 계산하여 시장의 위험도(Risk) 정보를 모델에 제공했습니다.

### 2.3. MLP 아키텍처 및 과적합 방지 (Deep Learning Strategy)
금융 데이터는 노이즈(Noise)가 매우 심한 데이터셋입니다. 따라서 딥러닝 모델이 훈련 데이터의 노이즈까지 학습해버리는 과적합 현상을 막는 것이 필수적입니다.

* **Model Architecture:** Input → FC(64) → ReLU → FC(64) → ReLU → FC(1) → Sigmoid
* **Dropout (0.1):** 학습 과정에서 무작위로 일부 뉴런을 비활성화하여 모델의 견고성(Robustness)을 높였습니다.
* **Early Stopping:** 매 Epoch마다 검증(Validation) 데이터셋의 손실(Loss)을 모니터링하여, 성능 개선이 20회 이상 멈출 경우 학습을 즉시 중단하고 최적의 가중치를 복구하는 로직을 구현했습니다.

---

## 3. 시스템 설계 및 구현 (Implementation)

전체 시스템은 유지보수와 확장이 용이하도록 객체 지향 프로그래밍(OOP) 방식으로 설계되었습니다.

### 3.1. 데이터 전처리 모듈: `build_features`
이동평균(MA)이나 모멘텀(Momentum)과 같은 기술적 지표들을 `pandas`의 벡터화 연산을 통해 고속으로 처리합니다. 타겟 변수 `y`는 `next_return > 0`일 경우 1(상승), 그렇지 않으면 0(하락/보합)으로 레이블링합니다.

```python
def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    
    # ... (로그 가격 및 수익률 계산 생략) ...

    # Moving Average Ratio 생성
    for w in cfg.ma_windows:
        ma = out["price"].rolling(w).mean()
        out[f"ma_ratio_{w}"] = out["price"] / ma - 1.0
    
    # Target Labeling: 내일 주가가 오르면 1, 아니면 0
    out["future_ret"] = out["ret"].shift(-1)
    out["y"] = (out["future_ret"] > 0).astype(int)
    return out.dropna().reset_index(drop=True)
```

### 3.2. 딥러닝 모델링: `TorchMLPModel`
PyTorch 프레임워크를 사용하여 심층 신경망을 구축하고 학습 프로세스를 제어합니다. 전체 데이터를 학습용(Train)과 검증용(Validation)으로 8:2 비율로 분리하며, 학습 중 검증 손실이 줄어들지 않으면 **Early Stopping**이 발동됩니다.

```python
class TorchMLPModel:
    # ... (초기화 코드 생략) ...
    def fit(self, X, y):
        # ... (데이터 분리 및 스케일링) ...
        for epoch in range(self.epochs):
            # ... (학습 및 검증 진행) ...
            
            # 조기 종료 (Early Stopping) 로직
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= self.patience:
                    print(f"*** Early Stopping at Epoch {epoch+1} ***")
                    break
        # 학습 종료 후 가장 성능이 좋았던 가중치로 복구
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
```

### 3.3. 백테스팅 시뮬레이터: `backtest`
단순히 예측값이 0.5를 넘는다고 매수하는 것이 아니라, **확률 임계값(Threshold)**을 두어 신중한 매매를 시뮬레이션했습니다. 상승 확률이 **60% 이상**일 때만 매수하고, **40% 이하**일 때만 매도합니다. 또한, 매 거래마다 0.05%의 수수료를 차감하여 현실성을 더했습니다.

```python
def backtest(prices, probs, cfg: BacktestConfig):
    # ... (자산 초기화) ...
    for price, p in zip(prices, probs):
        # Buy Signal: 강한 확신(0.6 이상)이 있을 때만 진입
        if shares == 0 and p >= cfg.p_buy:
            cost = cash * (cfg.commission + cfg.slippage)
            shares = (cash - cost) / price
            cash = 0
            
        # Sell Signal: 하락 확률이 높을 때(0.4 이하) 청산
        elif shares > 0 and p <= cfg.p_sell:
            cash = shares * price * (1 - cfg.commission - cfg.slippage)
            shares = 0
            
    # ... (최종 수익률, MDD 등 성과 지표 계산) ...
```

---

## 4. 실험 환경 및 결과 (Experiments & Results)

### 4.1. 실험 환경 및 로그 (Experimental Log)

* **대상 종목:** Apple (AAPL)
* **Train Period:** 2018-01-31 ~ 2022-11-30
* **Test Period:** 2022-12-01 ~ 2024-12-30

실제 모델 학습 과정에서 딥러닝 모델(MLP)은 **Early Stopping**이 작동하여 과적합을 방지했습니다.

```text
Downloading data for AAPL...

=== Train/Test Period ===
Train: 2018-01-31 ~ 2022-11-30
Test : 2022-12-01 ~ 2024-12-30

Training models...
 -> Logistic...
 -> RF...
 -> SVM...
 -> MLP...
*** Early Stopping at Epoch 23 (Best Val Loss: 0.6902) ***
```

### 4.2. 정량적 성과 비교 (Model Comparison)

실험 결과, 가장 단순한 모델인 **Logistic Regression**이 가장 높은 수익률을 기록했습니다.

**Benchmark (Buy&Hold) Return: 0.7184 (71.8%)**

| Model | Accuracy | CumRet (수익률) | MDD (최대낙폭) | Sharpe | Trades |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic** | **0.5421** | **0.8652 (86.5%)** | **-0.1661** | **1.5100** | **1** |
| RF | 0.5383 | 0.3226 (32.2%) | -0.1661 | 0.8363 | 1 |
| SVM | 0.5517 | 0.0000 (0.0%) | 0.0000 | 0.0000 | 0 |
| MLP | 0.5421 | 0.0000 (0.0%) | 0.0000 | 0.0000 | 0 |

### 4.3. 시각화 결과 (Visualizations)

#### (1) 자산 가치 변화 (Equity Curves)
<img width="1023" height="470" alt="image" src="https://github.com/user-attachments/assets/97ff2d40-b289-4fbc-b4c9-9589fb54b0d9" />
Logistic Regression(주황색)이 Benchmark(파란 점선)를 상회하며 우상향하는 모습을 확인할 수 있습니다.


#### (2) 낙폭 변화 (Drawdown Curves)
<img width="1030" height="393" alt="image" src="https://github.com/user-attachments/assets/373acb28-11bb-4b8f-8bd8-3dc13fb12cfe" />
대부분의 모델이 시장 하락기에 자산 가치가 감소하였으나, SVM과 MLP는 매매를 하지 않아 Drawdown이 0으로 유지되었습니다.


#### (3) 변수 중요도 (Feature Importance)
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/171c1807-2566-4aba-8e63-2902321dbb15" />
Random Forest 모델 분석 결과, `ret_lag_9`(9일 전 수익률)와 같은 과거의 추세 정보가 단기 변동성보다 예측에 더 중요한 영향을 미쳤습니다.

---

## 5. 결과 분석 및 고찰 (Discussion)

### 5.1. 단순함의 승리 (Occam's Razor)
실험 기간 동안 시장은 전반적인 **강세장 (Bull Market)** 이었습니다. 가장 단순한 선형 모델인 **Logistic Regression**은 이러한 상승 추세(Trend)를 빠르게 감지하여 초기에 매수한 후 계속 보유하는 전략을 취했습니다. 복잡한 패턴을 찾기보다는 전체적인 흐름에 편승하는 전략이 유효했으며, 결과적으로 86.5%라는 가장 높은 수익률을 기록했습니다.

### 5.2. 딥러닝 모델의 보수적 성향
반면, **MLP 모델**은 0%의 수익률을 기록했습니다. 이는 Early Stopping으로 인해 과적합은 방지되었으나, 시장의 노이즈 속에서 60% 이상의 높은 확신을 주는 패턴을 찾지 못했기 때문입니다. 즉, 예측 확률이 임계값(Threshold)을 넘지 못해 **"확실하지 않으면 투자하지 않는다"** 는 보수적인 결정을 내린 것으로 해석됩니다. 이는 손실을 보지 않았다는 점에서는 긍정적이나, 강세장에서의 기회비용을 잃었다는 한계가 있습니다.

### 5.3. Feature Importance의 시사점
Random Forest 분석 결과, 직전일의 수익률보다 `Lag 9` (약 2주 전)의 수익률이 더 중요한 변수로 선정되었습니다. 이는 주가 데이터에 단기적인 노이즈가 많아, 오히려 약간의 시차가 있는 과거 데이터가 추세 파악에 도움이 될 수 있음을 시사합니다.

---

## 6. 결론 (Conclusion)

본 프로젝트를 통해 **"복잡한 알고리즘이 항상 더 나은 수익을 보장하지 않는다"** 는 금융 데이터 분석의 중요한 교훈을 얻었습니다. 특히 노이즈가 심한 주가 데이터에서는 복잡한 딥러닝 모델이 오히려 과적합을 피하기 위해 지나치게 소극적으로 학습될 수 있음을 확인했습니다.

또한, 단순한 예측 정확도(Accuracy)보다는 **임계값 (Threshold) 설정**과 **수수료 (Cost) 관리**가 실제 포트폴리오 성과에 지대한 영향을 미친다는 것을 실증적으로 배웠습니다. 향후에는 **Grid Search**를 통한 최적의 매매 임계값 탐색과 **LSTM/Transformer**와 같은 시계열 특화 모델 도입을 통해 시스템을 고도화할 계획입니다.

---

## 🛠 Tech Stack

* **Language:** Python 3.10
* **Data Source:** `yfinance`
* **Deep Learning:** `torch`, `torch.nn`, `DataLoader`
* **Machine Learning:** `scikit-learn` (Logistic, SVM, RF)
* **Data Analysis:** `pandas`, `numpy`, `matplotlib`
