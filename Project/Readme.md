# 📈 기계학습 기반 주가 예측 및 자동매매 시뮬레이션 프로젝트
#### 기계학습과 응용 / 인형진 (2021131028)

---

## 1. 프로젝트 개요 및 모티베이션 (Project Motivation)

### 1.1. 배경 및 동기
저는 지난 학기 ‘수학과 프로그래밍’ 수업을 수강하며 점화식을 활용한 주가 예측 시뮬레이터를 개발한 경험이 있습니다. 당시 프로젝트는 수학적 모델을 코드로 구현하는 과정 자체에는 성공했으나, **랜덤하게 생성된 가상 데이터(Random Walk)**를 사용했다는 점과 단순한 **선형 점화식(Linear Recurrence)**에만 의존했다는 근본적인 한계가 있었습니다. 이는 실제 금융 시장의 복잡성과 불확실성을 반영하기에는 역부족이었습니다.

### 1.2. 프로젝트 목표
이번 학기 **<기계학습과 응용>** 수업을 통해, 실제 데이터는 훨씬 고차원적이고 비선형적인 패턴(Non-linear Pattern)을 내포하고 있음을 학습했습니다. 이에 본 프로젝트는 이전의 경험을 발전시켜 다음과 같은 심화된 목표를 설정하였습니다.

1.  **Real World Data Integration:** 가상의 데이터가 아닌 **Yahoo Finance API**를 연동하여, 실제 시장(Apple Inc.)의 Historical Data를 수집하고 전처리하는 파이프라인을 구축합니다.
2.  **Deep Learning Implementation:** 단순 회귀 분석을 넘어, **PyTorch 기반의 MLP(Multi-Layer Perceptron)** 모델을 직접 설계하고 학습시켜 딥러닝의 효용성을 검증합니다.
3.  **Robust Trading System:** 단순히 예측 정확도만 높이는 것이 아니라, **조기 종료(Early Stopping)** 기법을 통해 과적합을 제어하고, **거래 비용(Commission & Slippage)**을 반영한 현실적인 자동매매 시뮬레이션을 수행합니다.

---

## 2. 이론적 배경 (Theoretical Background)

본 프로젝트는 주식 시장의 예측 불가능성을 고려하여, 정확한 가격(Price)을 맞추는 회귀(Regression) 문제가 아닌, **등락 방향(Direction)을 예측하는 이진 분류(Binary Classification)** 문제로 접근하였습니다.

### 2.1. 피처 엔지니어링 (Feature Engineering)
단순히 '종가(Close Price)' 하나만으로는 모델이 유의미한 패턴을 학습하기 어렵습니다. 따라서 금융 도메인 지식을 활용하여 다음과 같은 파생 변수(Derived Features)를 생성하였습니다.
* **로그 수익률 (Log Returns):** 주가의 절대적인 크기에 영향을 받지 않도록 로그 차분(Log-Difference)을 사용하여 데이터의 정상성(Stationarity)을 확보했습니다.
* **지연 수익률 (Lagged Returns):** 시계열 데이터의 자기상관성(Autocorrelation)을 반영하기 위해 과거 1일~10일 전의 수익률을 입력 변수로 사용했습니다.
* **이동평균 괴리율 (MA Ratio):** 5일, 20일 이동평균선과 현재 주가 간의 거리를 측정하여 추세(Trend) 정보를 반영했습니다.
* **변동성 (Volatility):** 최근 가격의 표준편차를 계산하여 시장의 위험도(Risk) 정보를 모델에 제공했습니다.

### 2.2. MLP와 과적합 방지 (Overfitting Prevention)
금융 데이터는 노이즈(Noise)가 매우 심한 데이터셋입니다. 따라서 딥러닝 모델이 훈련 데이터의 노이즈까지 학습해버리는 과적합 현상을 막는 것이 필수적입니다.
* **Dropout:** 학습 과정에서 무작위로 일부 뉴런을 비활성화하여 모델의 견고성(Robustness)을 높였습니다.
* **Early Stopping:** 매 Epoch마다 검증(Validation) 데이터셋의 손실(Loss)을 모니터링하여, 성능 개선이 20회 이상 멈출 경우 학습을 즉시 중단하고 최적의 가중치를 복구하는 로직을 구현했습니다.

---

## 3. 시스템 설계 및 코드 구현 (Implementation)

전체 시스템은 유지보수와 확장이 용이하도록 객체 지향 프로그래밍(OOP) 방식으로 설계되었습니다.

### 3.1. 데이터 전처리 모듈: `build_features`

> **기능**: 수집된 원본 데이터를 머신러닝 모델 학습에 최적화된 텐서(Tensor) 형태로 변환합니다.

이동평균(Moving Average)이나 모멘텀(Momentum)과 같은 기술적 지표들을 `pandas`의 벡터화 연산을 통해 고속으로 처리합니다. 타겟 변수 `y`는 `next_return > 0`일 경우 1(상승), 그렇지 않으면 0(하락/보합)으로 레이블링합니다.

```python
def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()
    
    # ... (로그 가격 및 수익률 계산 생략) ...

    # 이동평균 괴리율 (Moving Average Ratio) 생성
    # 주가가 이동평균선보다 얼마나 위에 있는지(과매수), 아래에 있는지(과매도) 판단
    for w in cfg.ma_windows:
        ma = out["price"].rolling(w).mean()
        out[f"ma_ratio_{w}"] = out["price"] / ma - 1.0
    
    # Target Labeling: 내일 주가가 오르면 1, 아니면 0
    out["future_ret"] = out["ret"].shift(-1)
    out["y"] = (out["future_ret"] > 0).astype(int)
    return out.dropna().reset_index(drop=True)
```

### 3.2. 딥러닝 모델링: `TorchMLPModel`

> **기능**: PyTorch 프레임워크를 사용하여 심층 신경망을 구축하고 학습 프로세스를 제어합니다.

* **Network Architecture:** 입력층 → 은닉층(64) → 은닉층(64) → 출력층(1)의 3단계 구조를 가집니다. 활성화 함수로는 `ReLU`를 사용했습니다.
* **Training Process:** 전체 데이터를 학습용(Train)과 검증용(Validation)으로 8:2 비율로 분리합니다. 학습 중 검증 손실이 줄어들지 않으면 **Early Stopping**이 발동되어 과도한 학습을 방지합니다.

```python
class TorchMLPModel:
    # ... (초기화 코드 생략) ...
    def fit(self, X, y):
        # ... (데이터 분리 및 스케일링) ...
        for epoch in range(self.epochs):
            # ... (학습 및 검증 진행) ...
            
            # 조기 종료 (Early Stopping) 로직
            # 검증 손실이 최저점을 갱신하지 못하면 카운트를 증가시키고,
            # 인내심(patience) 한계에 도달하면 학습을 종료
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

> **기능**: 모델이 예측한 확률(Probability)을 기반으로 매수/매도 주문을 실행하고 포트폴리오 가치를 계산합니다.

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

## 4. 실험 결과 및 분석 (Experimental Results)

**실험 환경:** Apple (AAPL) 일별 데이터 (2018~2024년)  
**테스트 구간:** 2022년 12월 ~ 2024년 12월 (약 2년)

### 4.1. 정량적 성과 분석 (Model Comparison)

| Model | Accuracy | CumRet (수익률) | MDD (최대낙폭) | Sharpe | Trades (매매횟수) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic** | **0.5421** | **0.8652 (86.5%)** | **-0.1661** | **1.5100** | **1** |
| Benchmark | - | 0.7184 (71.8%) | -0.22XX | - | - |
| RF | 0.5383 | 0.3226 (32.2%) | -0.1661 | 0.8363 | 1 |
| SVM | 0.5517 | 0.0000 (0.0%) | 0.0000 | 0.0000 | 0 |
| MLP | 0.5421 | 0.0000 (0.0%) | 0.0000 | 0.0000 | 0 |

### 4.2. 자산 가치 변화 그래프 (Equity Curve)
<img width="1023" height="470" alt="image" src="https://github.com/user-attachments/assets/03835952-b6c9-4d8b-a2d9-1b91f1afffb2" />

*(위 그래프는 테스트 기간 동안 각 모델의 자산 가치 변화를 나타냅니다. 주황색 선인 Logistic Regression 모델이 파란색 점선인 Buy & Hold 벤치마크를 상회하는 것을 확인할 수 있습니다.)*

### 4.3. 변수 중요도 분석 (Feature Importance)
![Feature Importance](image_08d3c2.png)
*(Random Forest 모델이 추출한 변수 중요도입니다. 최근 1~2일의 변동보다 `ret_lag_9`(9일 전 수익률)와 같은 과거의 추세 정보가 예측에 더 중요한 영향을 미치는 것으로 나타났습니다.)*

### 4.4. 결과 심층 분석

1.  **단순함의 승리 (Occam's Razor):** 실험 기간 동안 시장은 전반적인 **강세장(Bull Market)**이었습니다. 가장 단순한 선형 모델인 **Logistic Regression**은 이러한 상승 추세(Trend)를 빠르게 감지하여 초기에 매수한 후 계속 보유하는 전략을 취했고, 결과적으로 86.5%라는 가장 높은 수익률을 기록했습니다.
2.  **딥러닝 모델의 보수적 성향:** 반면, **MLP 모델**은 0%의 수익률을 기록했습니다. 이는 Early Stopping으로 인해 과적합은 방지되었으나, 시장의 노이즈 속에서 60% 이상의 높은 확신을 주는 패턴을 찾지 못했기 때문입니다. 즉, 예측 확률이 임계값(Threshold)을 넘지 못해 **"확실하지 않으면 투자하지 않는다"**는 보수적인 결정을 내린 것으로 해석됩니다. 이는 손실을 보지 않았다는 점에서는 긍정적이나, 기회비용을 잃었다는 한계가 있습니다.

---

## 5. 결론 및 향후 개선 방향 (Conclusion)

### 5.1. 결론
이번 프로젝트를 통해 **"복잡한 알고리즘이 항상 더 나은 수익을 보장하지 않는다"**는 금융 데이터 분석의 중요한 교훈을 얻었습니다. 특히 노이즈가 심한 주가 데이터에서는 복잡한 딥러닝 모델이 오히려 과적합을 피하기 위해 지나치게 소극적으로 학습될 수 있음을 확인했습니다. 또한, 단순한 예측 정확도(Accuracy)보다는 **임계값(Threshold) 설정**과 **수수료(Cost) 관리**가 실제 포트폴리오 성과에 지대한 영향을 미친다는 것을 실증적으로 배웠습니다.

### 5.2. 개선 방향
* **Threshold 최적화 (Hyperparameter Tuning):** 현재 고정값으로 설정된 진입 장벽(0.6)이 너무 높았습니다. 향후 Grid Search 등을 통해 각 모델의 예측 성향에 맞는 최적의 매매 임계값을 동적으로 찾는 연구가 필요합니다.
* **포트폴리오 다각화 (Diversification):** 단일 종목(AAPL) 분석은 개별 기업의 이슈에 크게 휘둘릴 수 있습니다. S&P 500과 같은 지수(Index) 데이터나 다수 종목 포트폴리오를 대상으로 학습한다면 모델의 일반화 성능을 높일 수 있을 것입니다.
* **시계열 특화 모델 도입:** MLP 구조를 넘어, 시계열 데이터의 장기 의존성(Long-term Dependency)을 학습하는 데 특화된 **LSTM**이나 **Transformer** 아키텍처를 도입해보고 싶습니다.

---

## 6. 사용한 라이브러리 및 환경

본 프로젝트는 Python 3.x 환경에서 다음 라이브러리들을 활용하여 수행되었습니다.

* **Data Acquisition:** `yfinance` (Yahoo Finance API)
* **Data Processing:** `pandas`, `numpy`
* **Deep Learning:** `torch` (PyTorch), `torch.nn`, `DataLoader`
* **Machine Learning:** `scikit-learn` (Logistic, SVM, RF, Metrics)
* **Visualization:** `matplotlib.pyplot`

```python
# 필수 라이브러리 로드 예시
import yfinance as yf
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
```
