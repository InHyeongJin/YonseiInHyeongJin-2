# 📈 기계학습 기반 주가 예측 및 자동매매 시뮬레이션 프로젝트
#### 기계학습과 응용 / [본인 이름] ([본인 학번])

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
    # ... (중략) ...
    # Moving Average Ratio 생성
    for w in cfg.ma_windows:
        ma = out["price"].rolling(w).mean()
        out[f"ma_ratio_{w}"] = out["price"] / ma - 1.0
    
    # Target: Next Day Return > 0 (내일 오르면 1, 아니면 0)
    out["future_ret"] = out["ret"].shift(-1)
    out["y"] = (out["future_ret"] > 0).astype(int)
    return out.dropna().reset_index(drop=True)
```

