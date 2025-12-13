주제: PyTorch 기반 딥러닝을 활용한 주가 등락 예측 및 자동매매 시뮬레이션
기계학습과응용 / 인형진 (2021131028)
1. 모티베이션 (프로젝트를 하게 된 동기)
지난 학기 ‘수학과 프로그래밍’ 수업에서 선형 점화식을 이용한 주가 예측 프로젝트를 진행하며, 수학적 모델링이 실제 금융 데이터에 적용되는 과정에 큰 흥미를 느꼈습니다. 하지만 단순 선형 회귀 모델은 주가의 비선형적인 패턴과 복잡한 변동성을 담아내기에는 한계가 있음을 체감했습니다.

이번 ‘기계학습과 응용’ 수업을 통해 **딥러닝(Deep Learning)**과 분류(Classification) 기법을 배우면서, 기존의 선형 모델을 넘어선 고도화된 예측 시스템을 구축해보고 싶었습니다. 특히, 수업에서 배운 PyTorch를 활용하여 다층 퍼셉트론(MLP) 모델을 직접 구현하고, 이를 Scikit-learn의 전통적인 머신러닝 알고리즘들과 비교해 봄으로써 딥러닝의 효용성을 검증하고자 합니다. 또한, 단순 예측을 넘어 수수료와 슬리피지를 고려한 현실적인 백테스팅을 통해 실제 투자 전략으로서의 가능성을 타진해보고자 합니다.

2. 이론적 배경
지도 학습 기반의 분류 (Supervised Classification): 미래의 구체적인 가격을 맞추는 회귀(Regression) 대신, 내일 주가가 오를지(1) 내릴지(0)를 예측하는 분류 문제로 정의하여 예측의 정확도를 높인다.

다층 퍼셉트론 (Multi-Layer Perceptron, MLP): 입력층과 출력층 사이에 여러 개의 은닉층(Hidden Layer)을 두어 데이터의 비선형적인 특징을 학습하는 인공신경망 구조이다.

Feature Engineering (피처 엔지니어링): 단순히 과거 가격만을 사용하는 것이 아니라, 이동평균 괴리율(MA Ratio), 변동성(Volatility), 모멘텀(Momentum) 등 기술적 지표를 생성하여 모델의 학습 효율을 높인다.

백테스팅 지표:

Sharpe Ratio: 위험(변동성) 대비 수익률을 나타내는 지표.

MDD (Maximum Drawdown): 특정 기간 동안 겪을 수 있는 최대 자산 하락폭.

3. 코드 작성방법 및 설명
핵심 구성 요소: 데이터 수집부터 딥러닝 모델링, 백테스팅까지의 파이프라인
1. 데이터 수집 및 전처리 (load_price_yf, build_features)
역할: 실제 시장 데이터(AAPL)를 수집하고, 기계학습에 적합한 피처(Feature)를 생성합니다.

yfinance 라이브러리를 통해 2018년부터 2025년까지의 Apple(AAPL) 일별 주가 데이터를 다운로드합니다.

단순 가격 데이터 외에 학습에 필요한 파생 변수를 생성합니다:

lags: 과거 10일간의 수익률

ma_ratio: 5일, 20일 이동평균선 대비 현재가 비율

vol: 5일, 20일 변동성 (표준편차)

mom: 5일, 20일 전 대비 모멘텀

Target: 다음 날 주가가 오르면 1, 내리면 0으로 라벨링합니다.

Python

@dataclass
class FeatureConfig:
    lags: int = 10
    ma_windows: Tuple[int, int] = (5, 20)
    # ... (중략) ...

def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    # 로그 수익률, 이동평균, 변동성, 모멘텀 등 다양한 기술적 지표 생성
    # ...
    out["y"] = (out["future_ret"] > 0).astype(int) # Classification Target
    return out
2. PyTorch 기반 딥러닝 모델 (TorchMLP, TorchMLPModel)
역할: PyTorch의 nn.Module을 상속받아 MLP 모델을 구축하고 학습합니다.

구조: Input -> Linear(64) -> ReLU -> Dropout -> Linear(64) -> ReLU -> Dropout -> Output(1)

과적합 방지: Dropout(0.1)을 추가하여 학습 데이터에만 치중되는 현상을 방지했습니다.

학습 설정:

Optimizer: Adam

Loss Function: BCEWithLogitsLoss (이진 분류용 손실 함수)

Epochs: 200회 (충분한 학습을 위해 설정)

인터페이스: Scikit-learn 모델과 동일하게 fit(), predict_proba() 메서드를 구현하여 호환성을 확보했습니다.

Python

class TorchMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # 과적합 방지
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
# ... (학습 루프 및 예측 로직 포함)
3. 시뮬레이션 및 백테스팅 (BacktestConfig, backtest)
역할: 모델의 예측 확률을 기반으로 실제 매매를 수행하고 성과를 분석합니다.

매매 전략:

매수 조건: 상승 확률이 60% (p_buy) 이상일 때만 매수 (신중한 진입)

매도 조건: 상승 확률이 40% (p_sell) 이하로 떨어지면 매도

비용 반영: 매매 시 0.05%의 수수료(Commission)와 0.05%의 슬리피지(Slippage)를 차감하여 현실성을 높였습니다.

성과 측정: 누적 수익률(CumRet), 최대 낙폭(MDD), 샤프 지수(Sharpe)를 계산합니다.

Python

@dataclass
class BacktestConfig:
    p_buy: float = 0.60   # 신중한 매수 기준
    p_sell: float = 0.40  # 신중한 매도 기준
    commission: float = 0.0005
    slippage: float = 0.0005
    initial_cash: float = 10000.0
4. 실행 및 비교 (run)
역할: 전체 프로세스를 총괄하며 다양한 모델(Logistic, RF, SVM, MLP)과 벤치마크(Buy & Hold)를 비교합니다.

데이터를 Train(70%)과 Test(30%)로 분리합니다.

각 모델을 학습시키고 테스트 데이터에 대해 백테스팅을 수행합니다.

최종적으로 **수익률 곡선(Equity Curve)**과 낙폭 곡선(Drawdown Curve) 두 가지 그래프를 시각화합니다.

4. 프로젝트의 한계
단순 피드포워드 구조: 사용된 MLP 모델은 시계열 데이터의 '순서(Sequence)' 정보를 완벽하게 기억하지 못합니다. (LSTM이나 Transformer 대비 한계)

시장 국면의 변화: 2018~2022년 데이터로 학습했으나, 2023년 이후의 시장 트렌드(금리 인상 등)가 달라질 경우 예측력이 떨어질 수 있습니다.

수수료의 영향: 잦은 매매 신호가 발생할 경우, 모델의 정확도가 높아도 거래 비용으로 인해 실제 수익률은 낮아질 수 있음을 확인했습니다.

5. 실행 예시
[모델 성능 비교표]
(실제 실행 결과)

Plaintext

=== Model Comparison Table ===
Benchmark (Buy&Hold) Return: 0.XX (시장 수익률)

      Model     ACC      CumRet     MDD    Sharpe   Trades
4       MLP  0.5249     -0.1849  -0.2057  -0.8758       XX
3       SVM  0.5077     -0.1296  -0.1743  -0.5431       XX
...
(PyTorch MLP 모델이 정확도(ACC) 면에서 우수한 성능을 보였으나, 잦은 거래로 인해 수익률 방어에 과제가 남음)

[자산 가치 변화 (Equity Curve)]
모델별 수익률 변화를 벤치마크(Buy & Hold, 점선)와 비교한 그래프입니다.

[낙폭 변화 (Drawdown Curve)]
각 전략이 겪은 최대 손실폭을 시각화한 그래프입니다.

6. 결론 및 개선점
이번 프로젝트는 단순한 수학적 점화식을 넘어, PyTorch를 활용한 딥러닝 모델을 금융 데이터에 직접 적용해 보았다는 데 큰 의의가 있습니다. 특히, Feature Engineering을 통해 다양한 시장 데이터를 학습시키고, Dropout 등을 통해 과적합을 제어하려는 시도를 했습니다.

실험 결과, **MLP 모델이 전통적인 머신러닝 모델보다 높은 예측 정확도(Accuracy)**를 기록하며 딥러닝의 가능성을 보여주었습니다. 하지만 엄격한 수수료를 적용한 백테스팅 환경에서는 'Buy & Hold' 전략을 이기는 것이 쉽지 않음을 확인했습니다. 이는 단순한 방향성 예측을 넘어, **'언제 쉬어야 하는지'**를 아는 리스크 관리의 중요성을 시사합니다.

향후 발전 방향은 다음과 같습니다:

시계열 특화 모델 도입: MLP 대신 **LSTM(Long Short-Term Memory)**이나 GRU 모델을 적용하여 시계열 데이터의 장기 의존성을 학습시킵니다.

하이퍼파라미터 튜닝: 학습률(Learning Rate), 은닉층의 크기 등을 Grid Search로 최적화하여 성능을 극대화합니다.

앙상블 전략: MLP의 예측값과 Random Forest의 변수 중요도를 결합한 앙상블 모델을 구축하여 안정성을 높입니다.

지난 학기 프로젝트가 '구현'에 초점을 맞췄다면, 이번 학기는 **'데이터 기반의 의사결정'과 '딥러닝의 실무적 적용'**을 경험하는 값진 시간이었습니다.

사용한 라이브러리
Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
