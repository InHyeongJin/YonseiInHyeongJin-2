Markdown# 📈 PyTorch 기반 딥러닝을 활용한 주가 등락 예측 및 자동매매 시뮬레이션

#### **프로젝트 개요**
* **주제**: 기술적 지표를 활용한 주가 방향성 예측 (Classification) 및 시뮬레이션 기반 투자 전략 검증
* **핵심 기술**: PyTorch (MLP), Scikit-learn, 백테스팅, 데이터 시각화
* **작성자**: 인형진 (2021131028)

---

## 1. 🚀 모티베이션 (프로젝트 동기)

지난 프로젝트에서 단순 선형 모델의 한계를 경험한 후, 복잡한 주가 패턴을 학습하기 위해 **PyTorch 기반의 딥러닝(MLP)** 모델을 도입했습니다. 본 프로젝트의 목표는 단순한 예측 정확도(Accuracy)를 넘어, 현실적인 거래 비용(수수료, 슬리피지)을 반영한 **백테스팅**을 통해 딥러닝 모델의 실제 투자 효용성(Sharpe Ratio, MDD)을 검증하는 것입니다. 기존 머신러닝 모델(RF, SVM, Logistic)과 벤치마크(Buy & Hold) 대비 성능을 비교 분석하여 딥러닝 기반 자동매매 전략의 가능성을 탐색했습니다.

---

## 2. 💡 이론적 배경

| 분야 | 핵심 개념 | 적용 내용 |
| :--- | :--- | :--- |
| **문제 정의** | 지도 학습 기반 분류 (Classification) | 다음 날 주가가 **상승(1)**할지 **하락(0)**할지를 예측합니다. |
| **모델** | 다층 퍼셉트론 (MLP) | PyTorch를 사용하여 비선형적인 특징을 학습하는 심층 신경망을 구현했습니다. |
| **데이터** | 피처 엔지니어링 (Feature Engineering) | 과거 수익률, 이동평균 괴리율, 변동성, 모멘텀 등의 **기술적 지표**를 피처로 사용했습니다. |
| **평가 지표** | 샤프 지수 (Sharpe Ratio) | 위험(변동성) 대비 초과 수익률을 측정하여, 전략의 효율성을 평가합니다. |
| **평가 지표** | 최대 낙폭 (MDD) | 최악의 자산 하락 시나리오를 측정하여 리스크 관리 능력을 평가합니다. |

---

## 3. 💻 코드 작성 방법 및 주요 기능 설명

본 프로젝트는 데이터 수집부터 모델 학습, 백테스팅 및 시각화까지 일련의 파이프라인으로 구성되어 있습니다.

### 3.1. PyTorch 기반 딥러닝 모델 (`TorchMLPModel`)

모델의 깊이를 높이고 학습 시간을 늘려 예측 성능을 극대화했습니다.

* **구조**: `Linear(64) -> ReLU -> Dropout(0.1)` 2개 층을 쌓은 후 최종 출력층으로 구성.
* **학습 강화**: `epochs`를 **200**으로 설정하여 충분히 학습하도록 했습니다.
* **과적합 방지**: 각 은닉층 뒤에 `Dropout(0.1)`을 추가하여 일반화 성능을 개선했습니다.

```python
class TorchMLP(nn.Module):
    # ... (생략) ...
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            # ... (Layer 1) ...
            nn.Dropout(0.1),  # 과적합 방지
            # ... (Layer 2) ...
            nn.Dropout(0.1),
            # ... (Output) ...
        )
3.2. 현실적인 백테스팅 환경 (backtest)단순히 예측이 맞았는지 여부가 아닌, 실제 수익률을 기준으로 성능을 측정합니다.매매 기준 강화:p_buy (매수 확신): 0.60 (60%)으로 상향 조정하여 신중한 거래 유도p_sell (매도 기준): 0.40 (40%)으로 하향 조정거래 비용 반영: commission (수수료)와 slippage (슬리피지)를 각각 **0.0005 (0.05%)**로 적용하여 현실적인 순자산 변화를 계산했습니다.Python@dataclass
class BacktestConfig:
    p_buy: float = 0.60   # 신중한 매수 기준
    p_sell: float = 0.40  # 신중한 매도 기준
    commission: float = 0.0005 # 현실적 수수료 반영
    # ... (생략) ...
3.3. 모델 비교 및 벤치마크비교 모델: Logistic Regression, Random Forest (min_samples_leaf=10로 과적합 방지), SVM, PyTorch MLP벤치마크: Buy & Hold (매수 후 보유) 전략의 수익률과 비교하여 모델의 초과 수익 달성 여부를 평가합니다.4. 📉 프로젝트의 한계 및 과제MLP의 시계열적 한계: MLP는 시계열 데이터의 장기적인 의존성(Long-term dependency) 학습에 불리합니다. (LSTM/GRU 대비)정적 임계값: 매매 기준(p_buy=0.60)이 고정되어 있어 시장 상황에 따른 유연한 대응이 어렵습니다. 변동성이 높을 때는 더욱 보수적으로, 낮을 때는 공격적으로 임계값을 조정하는 동적 전략이 필요합니다.단일 종목: AAPL 단일 종목에 대한 최적화이므로, 범용적인 포트폴리오 전략으로 확장하기 위해서는 추가적인 연구가 필요합니다.5. 📊 실행 결과 (Execution Example)5.1. 모델 성과 비교표ModelACCCumRetMDDSharpeTradesBenchmark (Buy&Hold)N/A0.XX0.YY0.ZZ0MLP0.5249............RF...............SVM...............Logistic...............5.2. 시각화 결과1) 수익률 곡선 (Equity Curves)목표: 모델이 Buy & Hold 벤치마크(점선)를 상회하는지 확인결과:여기에 Equity Curve 그래프 이미지를 넣어주세요.2) 최대 낙폭 곡선 (Drawdown Curves)목표: 모델이 시장 하락기에 얼마나 자산을 잘 방어했는지 확인결과:여기에 Drawdown Curve 그래프 이미지를 넣어주세요.6. 🛠️ 결론 및 개선 방향PyTorch MLP 모델은 전통적인 ML 모델 대비 높은 방향성 예측 정확도를 보였으나, 잦은 거래 비용과 시장 변화에 대한 민감성으로 인해 샤프 지수(Sharpe Ratio) 면에서는 Buy & Hold를 압도하지 못하는 결과를 보였습니다.개선점시계열 특화 모델 도입: LSTM 또는 Transformer 기반의 모델을 적용하여 시계열 데이터의 시간적 특성을 더 잘 학습하도록 개선.동적 매매 임계값: 시장의 변동성(VIX 등)이나 모델의 불확실성을 반영하여 p_buy, p_sell 임계값을 동적으로 변경하는 전략 도입.손절매(Stop-Loss) 최적화: MDD를 줄이기 위한 효과적인 자산 방어 로직을 추가.하이퍼파라미터 최적화: Grid Search나 Bayesian Optimization을 통해 MLP의 은닉층 크기 및 학습률을 튜닝하여 성능 개선.7. 📚 사용한 라이브러리Pythonimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
