#  주제: 주가 예측 기반 자동매매 시뮬레이션 프로젝트
 
#### 수학과프로그래밍 / 인형진 (2021131028)

---

## 1. 모티베이션 (프로젝트를 하게 된 동기)

저는 약 3년간 해외 주식 투자를 경험하면서 주가의 변동성과 예측의 어려움을 직접 체감해 왔습니다. 이번 학기 ‘수학과 프로그래밍’ 수업에서 배운 파이썬의 데이터 시각화 기법(Matplotlib)이 특히 흥미로웠고, 이는 제가 직접 주가 그래프를 구현해 보고 싶다는 생각으로 이어졌습니다. 또한, 반복되는 수열 구조를 분석하는 점화식 개념 역시 매우 인상 깊었습니다. 이러한 요소들을 실제 투자 시뮬레이션에 접목시켜 보고자, 점화식을 활용한 단순한 주가 예측 모델을 설계하고 이를 기반으로 자동 매매 전략을 구현한 뒤, 시각화까지 포함한 통합적인 프로젝트를 진행하게 되었습니다.

---

## 2. 이론적 배경

* **선형 회귀**와 **이동 평균선 (Moving Average)**: 주가 데이터의 추세를 파악하는 기법으로 사용되며, 단기-장기 이동 평균선을 통해 매수·매도 시점을 파악할 수 있다.
* **선형 점화식 (Linear Recurrence Relation)**: 이전 항의 값으로 다음 항을 예측하는 수학적 모델로, 주가 예측 모델의 근간이 된다.
* **시뮬레이션 기반 자동매매 전략**: 일정한 규칙에 따라 매수·매도 조건을 판단하고 포지션을 취하는 전략으로, 인간의 개입 없이 자동으로 작동한다.

---

## 3. 코드 작성방법 및 설명

### 핵심 구성 요소: 총 4개의 클래스 기반 구조

---

### 1. `AdaptiveRecurrencePredictor`

> **역할**: 직전 두 시점의 가격을 기반으로 다음 시점 가격을 예측하는 단순 점화식 기반 선형 회귀 모델

* `__init__`: 예측 계수를 저장할 변수 `self.coefficients`를 초기화합니다.
* `fit(prev2, prev1, current)`: 두 시점의 과거 가격을 이용해 최소제곱법으로 선형 회귀 계수를 계산합니다.
* `predict_next(prev2, prev1)`: 학습된 계수를 이용해 다음 가격을 예측합니다.

```python
class AdaptiveRecurrencePredictor:
    def __init__(self):
        self.coefficients = None

    def fit(self, prev2, prev1, current):
        A = np.array([[prev2, prev1]])
        y = np.array([current])
        coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        self.coefficients = coeffs.flatten()
        return self.coefficients

    def predict_next(self, prev2, prev1):
        if self.coefficients is None:
            return prev1
        return float(self.coefficients[0] * prev2 + self.coefficients[1] * prev1)
2. MarketSimulator
역할: 랜덤 요인을 포함한 가격 흐름을 생성하여 투자 시뮬레이션 환경을 제공

초기 가격을 설정한 뒤, 추세 요인과 확률적 노이즈를 결합하여 가격을 생성합니다.

python
코드 복사
class MarketSimulator:
    def __init__(self, length):
        self.length = length
        self.prices = self._generate_prices()

    def _generate_prices(self):
        prices = [100, 102]
        for _ in range(self.length - 2):
            trend_factor = np.random.uniform(0.95, 1.05)
            shock = np.random.normal(0, 3)
            prices.append(max(1, trend_factor * prices[-1] + shock))
        return prices
3. Trader
역할: 예측값과 실제 가격의 차이를 기반으로 매수·매도·보유 결정을 수행

예측 상승폭에 따라 매수 비중을 조절합니다.

가격이 평균 매입가 대비 일정 비율 이상 하락하면 손절합니다.

거래 결과는 히스토리로 저장됩니다.

python
코드 복사
class Trader:
    def __init__(self, name, cash):
        self.name = name
        self.cash = cash
        self.shares = 0
        self.history = []
        self.avg_buy_price = None

    def decide(self, current_price, predicted_price):
        gain_ratio = (predicted_price - current_price) / current_price
        if gain_ratio > 0.02:
            to_buy = int(0.75 * self.cash / current_price)
            self.cash -= to_buy * current_price
            self.shares += to_buy
            self.avg_buy_price = current_price
            action = f"Buy {to_buy}"
        elif gain_ratio < -0.01 and self.shares > 0:
            self.cash += self.shares * current_price
            action = f"Sell {self.shares}"
            self.shares = 0
            self.avg_buy_price = None
        else:
            action = "Hold"
        self.history.append((current_price, predicted_price, self.cash, self.shares, action))
4. Simulator
역할: 전체 시뮬레이션을 실행하고 결과를 시각화

가격 생성 → 예측 → 매매 → 결과 출력 과정을 반복 수행합니다.

최종 자산 가치와 가격·자산 변화 그래프를 출력합니다.

python
코드 복사
class Simulator:
    def __init__(self, length=30):
        self.market = MarketSimulator(length)
        self.predictor = AdaptiveRecurrencePredictor()
        self.trader = Trader("Player", 10000)

    def run(self):
        prices = self.market.prices
        for i in range(2, len(prices)):
            pred = self.predictor.predict_next(prices[i-2], prices[i-1])
            self.trader.decide(prices[i], pred)
            self.predictor.fit(prices[i-2], prices[i-1], prices[i])
4. 프로젝트의 한계
단순한 선형 점화식 기반 예측으로 실제 주가의 비선형성을 충분히 반영하지 못함

거래 비용, 시장 유동성, 외부 변수 등은 고려되지 않음

규칙 기반 전략으로 복잡한 시장 상황 대응에는 한계가 있음

5. 실행 예시
(표 및 그래프 생략)

6. 결론 및 개선점
본 프로젝트를 통해 점화식 기반 예측 모델과 자동매매 전략을 결합한 주가 시뮬레이션을 구현해볼 수 있었습니다. 향후에는 기계학습 기반 예측 모델, 실제 주가 데이터, 거래비용을 포함한 보다 현실적인 시장 환경으로 확장해보고 싶습니다.

사용한 라이브러리
python
코드 복사
import numpy as np
import matplotlib.pyplot as plt
import random
