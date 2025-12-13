#  주제: 기계학습 기반 주가 방향 예측 및 거래비용을 고려한 자동매매 백테스트 프로젝트
 
#### 기계학습과응용 / 인형진 (2021131028)

---

## 1. 모티베이션 (프로젝트를 하게 된 동기)

저는 이전 학기에 ‘수학과프로그래밍’ 수업에서 주가 예측 기반 자동매매 시뮬레이션 프로젝트를 진행하며, 주가 데이터를 코드로 생성·시각화하고 간단한 예측 규칙을 적용해보는 경험을 했습니다. 그러나 해당 프로젝트는 시뮬레이션 데이터와 단순한 점화식 기반 예측 구조에 머물러 있어, 실제 금융 데이터에서의 성능이나 다양한 예측 모델 간 비교까지는 다루지 못했습니다.

이번 학기 ‘기계학습과응용’ 수업에서는 Logistic Regression, Decision Tree, Random Forest, SVM, 그리고 PyTorch 기반의 신경망 모델 등 다양한 기계학습 기법을 학습하였습니다. 이에 따라 이전 프로젝트를 발전시켜, 실제 주가 데이터를 기반으로 시계열 feature를 설계하고, 여러 기계학습 모델을 동일한 조건에서 비교하며, 예측 성능뿐 아니라 거래비용을 포함한 투자 성과까지 함께 분석하는 프로젝트를 진행하게 되었습니다.

---

## 2. 이론적 배경

* **이진 분류 (Binary Classification)**  
  다음 시점의 주가 수익률이 양수인지 여부를 예측하는 문제로 설정하여, 상승(1) / 하락(0) 형태의 분류 문제로 모델을 학습합니다.

* **시계열 Feature Engineering**  
  로그수익률, 과거 수익률 지연(lag), 이동평균 대비 비율(MA ratio), 변동성(volatility), 모멘텀(momentum) 등의 기술적 지표를 사용해 입력 변수를 구성합니다.

* **다중 기계학습 모델 비교**  
  Logistic Regression, Decision Tree, Random Forest, SVM, MLP(Pytorch)를 동일한 데이터와 feature로 학습시켜 공정한 비교를 수행합니다.

* **거래비용을 고려한 자동매매 시뮬레이션**  
  실제 투자 환경을 반영하기 위해 매수·매도 시 수수료(commission)와 슬리피지(slippage)를 포함한 백테스트를 수행합니다.

---

## 3. 코드 작성방법 및 설명

### 핵심 구성 요소: 데이터 로딩 → Feature 생성 → 모델 학습 → 자동매매 백테스트 → 결과 시각화

---

### 1. `load_price_yf`

> **역할**: 실제 주가 데이터를 불러와 분석에 적합한 형태로 전처리

* Yahoo Finance의 데이터를 `yfinance` 라이브러리를 통해 불러옵니다.
* 자동 주가 조정(auto_adjust=True)을 사용하여 배당·분할 효과를 반영합니다.
* 날짜(date)와 종가(price)만을 사용하여 DataFrame을 구성합니다.

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
역할: 주가 시계열로부터 기계학습 입력 변수(feature)와 타깃 변수(target)를 생성

로그가격 및 로그수익률 계산

과거 수익률 지연 변수(ret_lag)

이동평균 대비 비율(ma_ratio)

변동성(volatility)

모멘텀(momentum)

다음 시점 수익률의 부호를 기준으로 타깃 변수 y 생성

python
코드 복사
@dataclass
class FeatureConfig:
    lags: int = 10
    ma_windows: Tuple[int, int] = (5, 20)
    vol_windows: Tuple[int, int] = (5, 20)
    mom_windows: Tuple[int, int] = (5, 20)
3. TorchMLP / TorchMLPModel
역할: PyTorch 기반 다층 퍼셉트론(MLP)을 이용한 이진 분류 모델 구현

입력 feature를 표준화(StandardScaler)한 뒤 신경망에 입력합니다.

BCEWithLogitsLoss를 사용해 이진 분류 문제를 학습합니다.

sigmoid 함수를 통해 주가 상승 확률 P(up)을 출력합니다.

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
4. BacktestConfig / backtest
역할: 예측 확률을 기반으로 자동매매 전략을 수행하고 투자 성과를 계산

매수 조건: P(up) ≥ 0.55

매도 조건: P(up) ≤ 0.45

매수·매도 시 거래비용(commission, slippage) 반영

누적 수익률(Cumulative Return), 최대 낙폭(MDD), Sharpe Ratio 계산

python
코드 복사
@dataclass
class BacktestConfig:
    p_buy: float = 0.55
    p_sell: float = 0.45
    commission: float = 0.0005
    slippage: float = 0.0005
    initial_cash: float = 10000.0
5. run
역할: 전체 실험 파이프라인 실행

시계열 분할을 통해 Train/Test 기간을 명확히 구분합니다.

각 기계학습 모델에 대해 예측 성능(ACC, F1, AUC)을 계산합니다.

동일한 예측 결과를 자동매매 전략에 적용하여 투자 성과를 비교합니다.

결과를 표(DataFrame)와 그래프로 시각화합니다.

python
코드 복사
run()
4. 프로젝트의 한계
기술적 지표 중심의 feature만 사용하여 거시경제 변수나 뉴스 정보는 반영하지 못함

단일 종목(AAPL)에 한정된 분석으로 일반화에는 한계가 있음

단순 Long/Flat 전략으로 포지션 사이징 및 고급 리스크 관리는 포함하지 않음

거래비용을 고정값으로 가정하여 실제 시장의 복잡한 체결 구조는 완전히 반영하지 못함

5. 실행 방법 (Colab 기준)
python
코드 복사
# 라이브러리 설치
!pip -q install yfinance

# 코드 실행
run()
6. 결론 및 개선점
본 프로젝트에서는 실제 주가 데이터를 대상으로 시계열 기반 feature를 설계하고, 여러 기계학습 모델을 동일한 조건에서 비교하였습니다. 또한 예측 정확도뿐 아니라 거래비용을 포함한 백테스트 성과를 함께 분석함으로써, 예측 성능과 실제 투자 성과가 반드시 일치하지는 않는다는 점을 확인할 수 있었습니다.

향후에는 다중 종목 포트폴리오로 확장하거나, 포지션 사이징 및 리스크 관리 전략을 추가하고, LSTM이나 Transformer와 같은 시계열 딥러닝 모델을 적용하여 보다 현실적인 자동매매 시스템으로 발전시킬 수 있을 것입니다.

사용한 라이브러리
python
코드 복사
numpy
pandas
matplotlib
scikit-learn
torch
yfinance
markdown
코드 복사

---
