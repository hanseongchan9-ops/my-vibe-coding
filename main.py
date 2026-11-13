import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 페이지 기본 설정
# -----------------------------
st.set_page_config(
    page_title="Smart Energy Predictor",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "energydata_complete.csv"

FEATURES = [
    "T1", "RH_1",
    "T2", "RH_2",
    "T_out", "RH_out",
    "Windspeed", "lights",
    "hour_sin", "hour_cos",
    "day_sin", "day_cos"
]

# -----------------------------
# 모델 클래스 정의 (NumPy 기반)
# -----------------------------
class LinearRegressionModel:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        self.coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        return X_with_intercept @ self.coefficients

class MeanBaselineModel:
    def __init__(self):
        self.mean_value = None

    def fit(self, X, y):
        self.mean_value = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_value, dtype=float)

# -----------------------------
# 데이터 로드 및 전처리
# -----------------------------
@st.cache_data
def load_and_prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("energydata_complete.csv 파일을 찾을 수 없습니다. app.py와 같은 폴더에 업로드해 주세요.")

    raw_df = pd.read_csv(DATA_PATH)
    required = ["date", "Appliances", "T1", "RH_1", "T2", "RH_2", "T_out", "RH_out", "Windspeed", "lights"]

    raw_df["date"] = pd.to_datetime(raw_df["date"], errors="coerce")
    for col in required:
        if col != "date":
            raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    raw_df = raw_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    raw_df = raw_df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    df = raw_df.copy()
    
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    target_df = df[["date", "Appliances"]].rename(
        columns={"date": "target_time", "Appliances": "target_next"}
    )
    df["target_time"] = df["date"] + pd.Timedelta(minutes=10)
    df = df.merge(target_df, on="target_time", how="inner").drop(columns=["target_time"])
    
    df = df.dropna(subset=FEATURES + ["target_next"]).reset_index(drop=True)

    X = df[FEATURES].astype(float)
    y = df["target_next"].astype(float)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    splits = {
        "X_train": X.iloc[:train_end],
        "y_train": y.iloc[:train_end],
        "X_valid": X.iloc[train_end:valid_end],
        "y_valid": y.iloc[train_end:valid_end],
        "X_test": X.iloc[valid_end:],
        "y_test": y.iloc[valid_end:],
    }
    return raw_df, df, splits

@st.cache_resource
def get_trained_models(_splits):
    model = LinearRegressionModel().fit(_splits["X_train"], _splits["y_train"])
    baseline = MeanBaselineModel().fit(_splits["X_train"], _splits["y_train"])
    return model, baseline

def evaluate_metrics(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    errors = actual - pred
    
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

# -----------------------------
# 메인 앱 UI 구성
# -----------------------------
st.title("⚡ Smart Energy Predictor")
st.caption("가정용 전자기기 에너지 소비량 분석 및 10분 후 소비량 예측 시스템")

try:
    raw_df, processed_df, splits = load_and_prepare_data()
    model, baseline = get_trained_models(splits)
except Exception as e:
    st.error(f"데이터 및 모델 초기화 오류: {e}")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 프로젝트 소개",
    "📊 1. 데이터 분석",
    "🔮 2. 에너지 예측",
    "📈 3. 모델 평가"
])

# =============================
# TAB 1: 프로젝트 소개
# =============================
with tab1:
    st.subheader("프로젝트 개요")
    st.markdown("""
    이 시스템은 실내외 온습도, 풍속, 조명 사용량, 시간 데이터를 활용하여
    **10분 후 가정용 전자기기(Appliances)의 에너지 소비량(Wh)**을 예측합니다.
    """)

    st.info(
        "**탐구 질문:** 실내외 환경 센서 데이터와 시간대 정보를 활용하면 "
        "10분 후 가정용 전자기기의 에너지 소비량을 얼마나 정확하게 예측할 수 있을까?"
    )

    st.divider()
    st.caption("Data Source: UCI Machine Learning Repository | Appliances Energy Prediction")

# =============================
# TAB 2: 데이터 분석
# =============================
with tab2:
    st.header("데이터 기본 정보")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("전체 데이터 수", f"{len(raw_df):,}개")
    c2.metric("입력 변수 수", f"{len(FEATURES)}개")
    c3.metric("측정 간격", "10분")
    st.caption(f"데이터 측정 기간: {raw_df['date'].min()} ~ {raw_df['date'].max()}")

    st.divider()
    st.subheader("데이터 미리보기 (상위 10행)")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.divider()
    st.subheader("주요 변수 기초 통계량")
    variable_info = {
        "T1": "T1 (주방 온도, °C)",
        "RH_1": "RH_1 (주방 습도, %)",
        "T2": "T2 (거실 온도, °C)",
        "RH_2": "RH_2 (거실 습도, %)",
        "T_out": "T_out (외부 온도, °C)",
        "RH_out": "RH_out (외부 습도, %)",
        "Windspeed": "Windspeed (풍속, m/s)",
        "lights": "lights (조명 사용량, Wh)",
        "Appliances": "Appliances (전자기기 사용량, Wh)"
    }
    info_df = raw_df[list(variable_info.keys())].describe().T
    info_df["설명"] = info_df.index.map(variable_info)
    
    st.dataframe(info_df, use_container_width=True)

    st.divider()
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("전자기기 에너지 소비량 분포")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(raw_df["Appliances"], bins=30, color="#4C78A8", edgecolor="black")
        ax1.set_xlabel("Energy Consumption (Wh)")
        ax1.set_ylabel("Frequency")
        ax1.set_title("Appliances Energy Distribution")
        st.pyplot(fig1)
        plt.close(fig1)

    with col_chart2:
        st.subheader("시간대별 평균 에너지 소비량")
        hourly = raw_df.groupby(raw_df["date"].dt.hour)["Appliances"].mean()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(hourly.index, hourly.values, marker="o", color="#F58518")
        ax2.set_xlabel("Hour of Day (0-23)")
        ax2.set_ylabel("Mean Consumption (Wh)")
        ax2.set_title("Hourly Average Energy Consumption")
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig2)
        plt.close(fig2)

    st.divider()
    st.subheader("변수 간 상관관계 (Correlation)")
    corr_df = raw_df[list(variable_info.keys())].corr()
    st.dataframe(corr_df.style.background_gradient(cmap="coolwarm"), use_container_width=True)
    st.caption("※ 1이나 -1에 가까울수록 강한 상관관계를 의미하며, 이 결과를 바탕으로 예측 모델의 독립 변수를 선정했습니다.")

# =============================
# TAB 3: 에너지 예측
# =============================
with tab3:
    st.header("10분 후 에너지 소비량 예측")
    st.write("현재 시점의 실내외 환경과 시간 정보를 입력한 뒤 예측 버튼을 누르세요.")

    col_in1, col_in2 = st.columns(2)
    
    with col_in1:
        st.subheader("실내 환경")
        t1 = st.number_input("T1 (주방 온도, °C)", value=22.0, step=0.5)
        rh1 = st.number_input("RH_1 (주방 습도, %)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
        t2 = st.number_input("T2 (거실 온도, °C)", value=21.0, step=0.5)
        rh2 = st.number_input("RH_2 (거실 습도, %)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)

    with col_in2:
        st.subheader("외부 환경 및 조명")
        tout = st.number_input("T_out (외부 온도, °C)", value=12.0, step=0.5)
        rhout = st.number_input("RH_out (외부 습도, %)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)
        windspeed = st.number_input("Windspeed (풍속, m/s)", min_value=0.0, value=3.0, step=0.5)
        lights = st.number_input("lights (조명 사용량, Wh)", min_value=0.0, value=10.0, step=10.0)

    st.divider()
    
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        hour = st.slider("현재 시각 (시)", min_value=0, max_value=23, value=18)
    with col_t2:
        day_names = ["월요일 (Mon)", "화요일 (Tue)", "수요일 (Wed)", "목요일 (Thu)", "금요일 (Fri)", "토요일 (Sat)", "일요일 (Sun)"]
        day_name = st.selectbox("요일", day_names, index=0)
        dayofweek = day_names.index(day_name)

    st.divider()
    
    if st.button("10분 후 에너지 소비량 예측하기", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "T1": t1,
            "RH_1": rh1,
            "T2": t2,
            "RH_2": rh2,
            "T_out": tout,
            "RH_out": rhout,
            "Windspeed": windspeed,
            "lights": lights,
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_sin": np.sin(2 * np.pi * dayofweek / 7),
            "day_cos": np.cos(2 * np.pi * dayofweek / 7),
        }], columns=FEATURES)

        pred = float(model.predict(input_df)[0])
        avg_val = baseline.mean_value

        st.subheader("예측 결과")
        st.metric(
            label="10분 후 예상 에너지 소비량",
            value=f"{pred:.2f} Wh",
            delta=f"전체 평균 대비 {pred - avg_val:+.2f} Wh"
        )

        if pred < 0:
            st.warning("음수 예측값이 산출되었습니다. 선형회귀 모델의 선형적 구조 한계로 인해 나타난 현상이므로 0 Wh에 가까운 대기전력 상태로 해석해야 합니다.")
        else:
            st.info("입력된 환경 변수들의 복합적인 영향을 선형회귀식에 대입하여 산출된 예상 수치입니다.")

# =============================
# TAB 4: 모델 평가
# =============================
with tab4:
    st.header("선형회귀 모델 성능 평가")

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_valid, y_valid = splits["X_valid"], splits["y_valid"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    c1, c2, c3 = st.columns(3)
    c1.metric("학습 데이터 (70%)", f"{len(X_train):,}개")
    c2.metric("검증 데이터 (15%)", f"{len(X_valid):,}개")
    c3.metric("평가 데이터 (15%)", f"{len(X_test):,}개")

    test_pred = model.predict(X_test)
    base_pred = baseline.predict(X_test)

    model_scores = evaluate_metrics(y_test, test_pred)
    base_scores = evaluate_metrics(y_test, base_pred)

    st.divider()
    st.subheader("평가 지표 요약 (평가 데이터 기준)")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{model_scores['MAE']:.2f} Wh")
    m2.metric("RMSE", f"{model_scores['RMSE']:.2f} Wh")
    m3.metric("R²", f"{model_scores['R2']:.4f}")

    comparison_df = pd.DataFrame({
        "평가 지표": ["MAE (Wh)", "RMSE (Wh)", "R2"],
        "평균 기준 모델": [
            round(base_scores["MAE"], 4),
            round(base_scores["RMSE"], 4),
            round(base_scores["R2"], 4)
        ],
        "선형회귀 모델": [
            round(model_scores["MAE"], 4),
            round(model_scores["RMSE"], 4),
            round(model_scores["R2"], 4)
        ]
    })
    st.dataframe(comparison_df, use_container_width=True)

    st.divider()
    st.subheader("실제값 vs 예측값 시계열 비교 (테스트 데이터 200개 구간)")
    
    chart_df = pd.DataFrame({
        "Actual (Wh)": y_test.values[:200],
        "Predicted (Wh)": test_pred[:200]
    })
    
    st.line_chart(chart_df)
    st.caption("※ 모델이 전반적인 흐름은 따라가지만, 갑작스러운 전력 소비 급증(스파이크) 구간은 선형 모델의 한계로 완벽히 포착하지 못함을 알 수 있습니다.")

    st.divider()
    st.subheader("각 환경 변수가 에너지 소비량에 미치는 영향 (회귀 계수)")
    
    coefs = model.coefficients[1:]
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": coefs
    }).sort_values(by="Coefficient", ascending=True)

    fig_coef, ax_coef = plt.subplots(figsize=(9, 6))
    colors = ['#4C78A8' if c < 0 else '#E45756' for c in coef_df["Coefficient"]]
    
    ax_coef.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors, edgecolor="black")
    ax_coef.axvline(0, color="gray", linestyle="--", linewidth=1)
    
    ax_coef.set_xlabel("Impact on Energy Consumption (Wh) per 1 unit change")
    ax_coef.set_title("Linear Regression Coefficients (Feature Importance)")
    st.pyplot(fig_coef)
    plt.close(fig_coef)

    st.caption("※ 양수(빨간색 막대) 변수는 값이 커질수록 전력 소비량을 늘리는 요인이며, 음수(파란색 막대) 변수는 줄이는 요인입니다. "
               "단위가 다른 변수들 간의 단순 크기 비교이므로 절대적 영향력 순위로 해석할 수는 없습니다.")

    st.divider()
    st.subheader("예측 오차(잔차) 분포")
    errors = y_test.values - test_pred
    
    fig3, ax3 = plt.subplots(figsize=(9, 3.5))
    ax3.hist(errors, bins=35, color="#54A24B", edgecolor="black")
    ax3.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax3.set_xlabel("Prediction Error (Actual - Predicted, Wh)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Residual Error Distribution")
    
    st.pyplot(fig3)
    plt.close(fig3)
