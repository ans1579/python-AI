import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import warnings
import time # 시뮬레이션을 위한 time.sleep 추가
import requests # API 호출을 위해 requests 라이브러리 추가
import os # API 키 로드를 위해 os 라이브러리 추가
from dotenv import load_dotenv

# 경고 무시
warnings.filterwarnings("ignore")

# 1. 파일 경로 설정 (Excel 파일 형식)
# NOTE: 이 경로는 실행 환경에 맞게 조정되어야 합니다.
FILE_GLOBAL_2024 = './2024_Global_Sales_Clean.xlsx' 
FILE_GLOBAL_2025 = './2025_Global_Sales_Clean.xlsx'
FILE_MODEL_2024 = './HMC-modelbyeol-panmae-2024nyeon-clean.xlsx' 
FILE_MODEL_2025 = './HMC-modelbyeol-panmae-2025nyeon-clean.xlsx' 

# Streamlit 앱 설정
st.set_page_config(layout="wide")
st.title("2026년 글로벌 자동차 '수출' 수요 예측 및 모델별 분석 보고서")
st.markdown("---")

# ---------------------------------------------
# 데이터 로드 함수 (pd.read_excel 사용)
# ---------------------------------------------
@st.cache_data
def load_and_process_global_sales(filepath, year):
    """글로벌 판매 데이터를 로드하고 '수출' 판매량만 월별로 집계합니다."""
    try:
        df = pd.read_excel(filepath) 
        month_cols = [col for col in df.columns if any(m in col for m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])]
        
        if 'Plant' not in df.columns:
            raise ValueError(f"'{filepath}'에 'Plant' 컬럼이 없습니다.")
            
        df['Plant'] = df['Plant'].astype(str)
        # '수출'이 포함된 Plant의 월별 판매량 합계
        export_sales = df[df['Plant'].str.contains('수출', na=False)][month_cols].sum()
        
        # 인덱스를 datetime 형식으로 변환
        month_names = [col.split('.')[0] for col in month_cols]
        export_sales.index = pd.to_datetime([f"{year}-{m}-01" for m in month_names], format="%Y-%b-%d")
        return export_sales.rename(f'Export_Sales_{year}')
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {filepath} - 오류: {e}")
        return pd.Series()

@st.cache_data
def load_and_process_model_sales(filepath, year):
    """모델별 판매 데이터를 로드하고 Total 판매량을 반환합니다."""
    try:
        df = pd.read_excel(filepath)
        
        if 'Model' not in df.columns or 'Total' not in df.columns:
            raise ValueError(f"'{filepath}'에 'Model' 또는 'Total' 컬럼이 없습니다.")
        
        df = df[['Model', 'Total']].set_index('Model')
        df.columns = [f'Total_Sales_{year}']
        return df
    except Exception as e:
        st.error(f"데이터 처리 중 오류 발생: {filepath} - 오류: {e}")
        return pd.DataFrame()
    
# ---------------------------------------------
# Gemini API 호출 함수 (실제 코드)
# ---------------------------------------------
def generate_ai_report_text(data_summary):
    """
    분석된 데이터를 기반으로 Gemini API에 요청을 보내 전략 보고서를 생성하는 함수입니다.
    네트워크 및 API 키 문제로 인해 실제 호출은 주석 처리하고 시뮬레이션 응답을 반환합니다. 
    (실제 사용 시에는 주석 해제 후 API 키 설정 필요)
    """
    # 1. System Instruction: Define the model's persona and rules
    system_instruction = "당신은 글로벌 자동차 시장 분석 전문가입니다. 주어진 데이터를 기반으로 알기쉽게 전략적인 행동 계획을 포함하는 보고서 요약을 한국어로 작성해야 합니다."
    
    # 2. User Prompt: The specific data and task (데이터를 JSON 형태로 직렬화하여 전달하는 것이 가장 이상적입니다.)
    user_prompt = f"""
    다음은 2024년과 2025년의 판매 데이터를 분석한 결과입니다:
    1. 2026년 총 예측 수출 판매량: {data_summary['total_forecast']:,}대 (월평균 {data_summary['monthly_avg']:,}대).
    2. 최고 성장 모델: {data_summary['top_model']} ({data_summary['top_growth_rate']:.2f}% 성장).
    3. 최대 하락 모델: {data_summary['bottom_model']} ({data_summary['bottom_decline_rate']:.2f}% 하락).
    
    이 데이터를 기반으로, 경영진을 위한 2026년 핵심 전략 3가지(수요 대응, 수익성 확보, 리스크 관리)를 요약하여 작성해 주세요.
    """
    
    # 3. Gemini API 호출 설정
    load_dotenv()
    
    API_KEY = os.environ.get("GEMINI_API_KEY")
    MODEL_NAME = "gemini-2.5-flash"
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    
    # API 요청 페이로드
    payload = {
        "contents": [{ "parts": [{ "text": user_prompt }] }],
        "systemInstruction": { "parts": [{ "text": system_instruction }] },
        # 검색 기반 Grounding이 필요한 경우 다음 줄을 추가합니다.
        "tools": [{ "google_search": {} }], 
    }

    ai_report_text = ""
    max_retries = 3
    retry_delay = 1 # seconds

    # 4. API 호출 (실제로는 주석 처리하고 시뮬레이션 응답 반환)
    
    for attempt in range(max_retries):
        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            response.raise_for_status() # HTTP 오류가 발생하면 예외를 발생시킵니다.
            result = response.json()
            
            # 응답에서 텍스트 추출
            ai_report_text = result['candidates'][0]['content']['parts'][0]['text']
            
            break # 성공하면 루프를 종료합니다.
        
        except requests.exceptions.RequestException as e:
            st.warning(f"API 호출 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt)) # Exponential backoff
            else:
                ai_report_text = "API 호출에 실패하여 보고서를 생성할 수 없습니다. 네트워크 연결 또는 API 키를 확인하십시오."
                

    # 5. 시뮬레이션 응답 (API 호출이 주석 처리되었으므로 이 응답을 반환합니다.)
    if not ai_report_text:
        ai_report_text = f"""
    ### 2026년 AI 기반 핵심 전략 요약 보고서 (시뮬레이션)
    
    **1. 수요 대응 및 예측 모델 고도화**
    2026년 전체 수출 수요는 약 **{data_summary['total_forecast']:,}대**로 안정적인 수준이 예상되므로, 생산 계획은 월평균 **{data_summary['monthly_avg']:,}대** 수준을 기본으로 설정합니다. 예측 불확실성을 최소화하기 위해 $\text{{SARIMA}}$ 모델이나 외부 변수를 포함하는 $\text{{ARIMAX}}$ 모델로의 전환을 통해 예측 정확도를 높이는 작업을 2분기까지 완료해야 합니다.
    
    **2. 수익성 중심 포트폴리오 강화**
    **{data_summary['top_model']}**의 **{data_summary['top_growth_rate']:.2f}%** 성장세는 고마진 프리미엄 $\text{{SUV}}$ 세그먼트의 강력한 수요를 방증합니다. 2026년에는 이들 고수익 모델의 수출 물량을 최우선으로 확보하고, 글로벌 시장에서 프리미엄 브랜드 이미지를 공고히 하기 위한 전략적 투자를 확대해야 합니다.
    
    **3. 주력 세단 및 $\text{{EV}}$ 리스크 긴급 진단**
    **{data_summary['bottom_model']}** 모델에서 **{data_summary['bottom_decline_rate']:.2f}%**의 급격한 판매 하락세가 관찰되었습니다. 이는 모델 경쟁력 약화 및 시장 점유율 손실로 직결될 수 있으므로, 해당 모델에 대한 국가별/지역별 하락 원인을 긴급 진단해야 합니다. 또한, $\text{{EV}}$ 모델의 판매 둔화 징후에 대응하여 재고 수준을 면밀히 모니터링하고 탄력적인 인센티브 정책을 준비해야 합니다.
    """
    
    return {
        'system_instruction': system_instruction,
        'user_prompt': user_prompt,
        'ai_report_text': ai_report_text.strip()
    }

# ---------------------------------------------
# 핵심 데이터 로드 및 ARIMA 모델 실행 (기존 코드)
# ---------------------------------------------
sales_2024 = load_and_process_global_sales(FILE_GLOBAL_2024, 2024)
sales_2025 = load_and_process_global_sales(FILE_GLOBAL_2025, 2025)

if sales_2024.empty or sales_2025.empty:
    st.error("필수 데이터 파일 로드에 실패했습니다. 파일 경로, 이름, Excel 시트 이름을 확인해 주세요.")
    st.stop()

combined_sales = pd.concat([sales_2024, sales_2025]).sort_index()

try:
    # ARIMA 모델 학습 및 예측
    model = ARIMA(combined_sales, order=(1, 1, 1), freq='MS') 
    model_fit = model.fit()

    forecast_steps = 12
    forecast = model_fit.get_forecast(steps=forecast_steps)
    forecast_mean = forecast.predicted_mean.round(0).astype(int)
    
    # 예측 결과 DataFrame 생성
    conf_int_df = forecast.conf_int()
    forecast_df = pd.DataFrame({
        'Predicted_Export_Sales (2026)': forecast_mean,
        'Lower Bound': conf_int_df.iloc[:, 0].round(0).astype(int),
        'Upper Bound': conf_int_df.iloc[:, 1].round(0).astype(int),
    })
    forecast_df.index = forecast_df.index.strftime('%Y-%m')

except Exception as e:
    st.error(f"ARIMA 모델 학습 중 오류 발생: {e}")
    st.caption("데이터가 불안정하거나 길이가 짧아 모델이 수렴하지 않을 수 있습니다.")
    st.stop()

# 모델별 판매 데이터 로드 및 성장률 계산
model_sales_2024 = load_and_process_model_sales(FILE_MODEL_2024, 2024)
model_sales_2025 = load_and_process_model_sales(FILE_MODEL_2025, 2025)

model_comparison = pd.merge(
    model_sales_2024, 
    model_sales_2025, 
    left_index=True, 
    right_index=True, 
    how='inner' 
)

model_comparison['Growth_Rate (%)'] = (
    (model_comparison['Total_Sales_2025'] - model_comparison['Total_Sales_2024']) / model_comparison['Total_Sales_2024']
) * 100

# 최소 판매량 필터링 및 정렬
model_comparison = model_comparison[
    (model_comparison['Total_Sales_2024'] >= 500) | 
    (model_comparison['Total_Sales_2025'] >= 500)
].sort_values(by='Growth_Rate (%)', ascending=False)


# ---------------------------------------------
# 📊 분석 보고서 생성 (기존 UI)
# ---------------------------------------------

st.header("1. 개요 및 주요 분석 결과 요약")

total_forecast = forecast_mean.sum()
monthly_avg = forecast_mean.mean()
top_growth_model = model_comparison.iloc[0].name
bottom_decline_model = model_comparison.iloc[-1].name

st.markdown("""
본 보고서는 2024-2025년 글로벌 자동차 수출 데이터를 기반으로 2026년 수요를 예측하고, 모델별 전년 대비 성장률을 비교하여 전략적 시사점을 도출합니다.
""")

col_metric_1, col_metric_2, col_metric_3 = st.columns(3)

with col_metric_1:
    st.metric(
        label="2026년 총 예측 수출 판매량", 
        value=f"{total_forecast:,.0f} Units",
        delta=f"월평균: {monthly_avg:,.0f} Units"
    )

with col_metric_2:
    st.metric(
        label="최고 성장 모델 (2025년 기준)", 
        value=top_growth_model, 
        delta=f"{model_comparison.iloc[0]['Growth_Rate (%)']:.2f}% 성장",
        delta_color="normal"
    )

with col_metric_3:
    st.metric(
        label="최대 하락 모델 (2025년 기준)", 
        value=bottom_decline_model, 
        delta=f"{model_comparison.iloc[-1]['Growth_Rate (%)']:.2f}% 하락",
        delta_color="inverse"
    )
    
st.markdown("---")

# ---------------------------------------------
# 2. 2026년 수출 수요 예측 분석
# ---------------------------------------------
st.header("2. 2026년 수출 수요 예측 분석 (ARIMA 모델 기반)")

col_2_1, col_2_2 = st.columns([1, 1])

with col_2_1:
    st.subheader("2.1. 2026년 월별 예측 결과 및 신뢰 구간")
    st.dataframe(forecast_df)
    st.caption("🚨 ARIMA(1,1,1) 모델은 단순 추세를 반영하며, 예측 기간이 길어질수록 신뢰 구간(Lower/Upper Bound)이 넓어져 불확실성이 증가합니다.")

with col_2_2:
    st.subheader("2.2. 전체 수출 판매량 추이 (2024-2026 예측)")
    
    # 시각화 데이터 준비
    plot_series = combined_sales.rename('Actual Sales').to_frame()
    forecast_plot = forecast_mean.rename('Actual Sales').to_frame()
    # 실제 값과 예측 값을 합치기 위해 임시로 예측 값의 Date 컬럼을 문자열로 통일
    full_plot_data = pd.concat([plot_series, forecast_plot]).reset_index()
    full_plot_data.columns = ['Date', 'Sales']
    full_plot_data['Type'] = ['Actual'] * len(combined_sales) + ['Forecast'] * len(forecast_mean)
    full_plot_data['Date'] = pd.to_datetime(full_plot_data['Date']).dt.strftime('%Y-%m')

    fig = px.line(
        full_plot_data, 
        x='Date', 
        y='Sales', 
        color='Type', 
        markers=True,
        title='전체 수출 판매량 추이 (2024-2026 예측)'
    )
    # 실제 데이터의 마지막 시점을 구분선으로 표시
    fig.add_vline(x=combined_sales.index[-1].strftime('%Y-%m'), line_width=1, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_tickangle=-45, legend_title_text='Data Type')
    st.plotly_chart(fig, use_container_width=True) # 

st.markdown("""
**시사점:**
* **안정적인 수요:** 예측 모델은 2026년에도 현재의 월별 평균 수준($4$만 대 내외)을 **큰 변동 없이 유지**할 것으로 전망합니다.
* **모델 고도화 필요:** 자동차 수요의 **계절성**을 반영하기 위해 $\text{SARIMA}$ 모델로 전환하거나, 글로벌 $\text{GDP}$ 등 **외부 변수($\text{ARIMAX}$)**를 고려하여 예측 정확도를 높이는 작업이 필요합니다.
""")

st.markdown("---")

# ---------------------------------------------
# 3. 모델별 판매 성장률 분석
# ---------------------------------------------
st.header("3. 모델별 판매 성장률 분석 (2024년 vs 2025년)")

st.subheader("3.1. 최고 성장률 Top 5 모델")

top_5 = model_comparison.head(5).style.format(
    {'Total_Sales_2024': '{:,.0f}', 'Total_Sales_2025': '{:,.0f}', 'Growth_Rate (%)': '{:.2f}%'}
)
st.dataframe(top_5, use_container_width=True)

st.markdown("""
* **프리미엄 및 $\text{SUV}$ 강세:** Top 5 성장 모델 중에는 $\text{GV80}$, $\text{G90}$ 등 고가 라인업의 성장이 두드러집니다. 이는 **수익성 높은 세그먼트**의 수요가 증가하고 있음을 시사합니다.
* **볼륨 리더:** $\text{Avante}$와 같은 볼륨 모델은 여전히 견고한 판매량을 유지하며 성장을 주도하고 있습니다.
""")


st.subheader("3.2.  최대 하락률 Bottom 5 모델")

# 성장률이 가장 낮은 5개 모델
bottom_5 = model_comparison.tail(5).style.format(
    {'Total_Sales_2024': '{:,.0f}', 'Total_Sales_2025': '{:,.0f}', 'Growth_Rate (%)': '{:.2f}%'}
)
st.dataframe(bottom_5, use_container_width=True) # 

st.markdown("""
* **경쟁력 약화 모델:** $\text{G70}$ 등 주력 세단 모델의 하락세는 **경쟁 모델 출시나 모델 노후화**에 따른 경쟁력 약화를 시사합니다.
* **$\text{EV}$ 재고 리스크:** $\text{GV70 EV}$의 하락세는 글로벌 $\text{EV}$ 시장의 일시적 둔화 또는 초기 물량 확보 이후의 재고 리스크에 대한 검토가 필요함을 의미합니다.
* **긴급 진단 필요:** `Grandeur`와 같은 대형 볼륨 모델의 하락($-35.13\%$는 데이터상 큰 폭의 하락임)은 긴급히 원인(국가별 판매 감소 등)을 진단해야 합니다.
""")

st.markdown("---")

# ---------------------------------------------
# 4. 종합 결론 및 전략적 제언 (기존 결론)
# ---------------------------------------------
st.header("4. 종합 결론 및 전략적 제언")

st.info(f"""
1.  **2026년 총 수요 대응:** 예측된 총 수출 수요 **{total_forecast:,.0f}대**를 목표로 생산 계획을 수립하고, 예측 불확실성을 낮추기 위한 모델 고도화(SARIMA, ARIMAX)를 추진해야 합니다.
2.  **수익성 중심 운영:** **{top_growth_model}** 등 **고마진 프리미엄 SUV 모델**의 수출 물량을 우선적으로 확보하고, 이들 모델의 성장 동력을 유지하기 위한 마케팅 투자를 강화해야 합니다.
3.  **리스크 관리:** 하락세가 뚜렷한 **주력 세단 모델({bottom_decline_model} 등)**에 대해서는 연식 변경, 프로모션 강화 등의 조치를 통해 2026년 경쟁력 회복을 모색해야 합니다.
""")

# ---------------------------------------------
# 5. 🤖 AI 기반 전략 보고서 생성 (Gemini 통합)
# ---------------------------------------------

# AI 보고서 생성을 위한 데이터 요약 준비
data_for_ai = {
    'total_forecast': total_forecast,
    'monthly_avg': int(monthly_avg),
    'top_model': top_growth_model,
    'top_growth_rate': model_comparison.iloc[0]['Growth_Rate (%)'],
    'bottom_model': bottom_decline_model,
    'bottom_decline_rate': model_comparison.iloc[-1]['Growth_Rate (%)']
}

st.markdown("---")
st.header("5. 🤖 AI 기반 전략 보고서 생성 (Gemini 통합)")
st.markdown("분석된 핵심 데이터를 기반으로 Gemini 모델에게 2026년 경영 전략 요약을 요청합니다.")

# Streamlit button to trigger the AI report generation
if st.button("🚀 Gemini에게 2026년 핵심 전략 보고서 요청", type="primary"):
    with st.spinner("Gemini가 데이터를 분석하고 전략 보고서를 작성 중입니다..."):
        
        # Call the simulation function
        ai_result = generate_ai_report_text(data_for_ai)
        
        st.subheader("Gemini Strategic Report")
        st.markdown(ai_result['ai_report_text'])
        
        st.caption("※ 이 보고서는 2024-2025년 판매 데이터 분석 결과를 바탕으로 Gemini 모델이 생성한 전략 요약본입니다.")
        
        # Gemini API 호출에 사용된 프롬프트 정보 표시 (투명성 확보)
        with st.expander("Gemini 모델에 전달된 프롬프트 정보 확인"):
            st.markdown("---")
            st.markdown("**System Instruction (AI 역할 정의)**")
            st.code(ai_result['system_instruction'], language='markdown')
            st.markdown("---")
            st.markdown("**User Prompt (AI에게 전달된 데이터 및 요청)**")
            st.code(ai_result['user_prompt'], language='markdown')

# ---------------------------------------------
# ---------------------------------------------