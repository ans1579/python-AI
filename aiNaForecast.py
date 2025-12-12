import pandas as pd
import streamlit as st
import base64
from io import BytesIO
import json
import time
import requests # API 통신을 위해 requests 모듈 사용 (환경에 따라 필요)
import os # API 키 로드를 위해 os 라이브러리 추가
from dotenv import load_dotenv

# --- 상수 및 설정 ---
load_dotenv()
ST_PAGE_TITLE = "2026년 북미 자동차 수요 예측 및 분석 대시보드"
GEMINI_MODEL = "gemini-2.5-flash"    
API_KEY = os.environ.get("GEMINI_API_KEY")

FILE_2023 = "./hmc-us-retail-sales-december-y2023.xlsx"
FILE_2024 = "./hmc-us-retail-sales-december-y2024.xlsx"
FILE_2025_JAN = "./hmc-us-retail-sales-january-y2025.xlsx"
ALL_FILES = [
    (FILE_2023, 2023),
    (FILE_2024, 2024),
    (FILE_2025_JAN, 2025)
]

# 월 이름 매핑
MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

# --- 데이터 로드 및 통합 함수 (시계열 분석을 위한 Long 포맷으로 변환) ---

def load_and_clean_data(filepath: str, year: int) -> pd.DataFrame:
    """
    엑셀 파일을 직접 읽고 월별 데이터를 Long 포맷으로 변환하여 반환합니다.
    """
    try:
        # 실제 헤더 행은 3번째 행 (인덱스 2)에 위치
        df = pd.read_excel(filepath, header=2) 
    except Exception as e:
        st.error(f"'{filepath}' 파일을 로드하는 중 심각한 오류 발생: {e}")
        return pd.DataFrame()

    df_clean = df.copy()
    df_clean.columns = [str(col).strip() for col in df_clean.columns]
    
    # 'Model' 컬럼 찾기 및 통일
    model_col_candidate = next((c for c in df_clean.columns if c.lower().strip() in ['model', 'models']), None)

    if not model_col_candidate:
        st.warning(f"'{filepath}' 파일에서 'Model' 컬럼을 찾을 수 없습니다. (현재 컬럼명: {df_clean.columns.tolist()})")
        return pd.DataFrame()
    
    df_clean.rename(columns={model_col_candidate: 'Model'}, inplace=True)
    model_idx = df_clean.columns.get_loc('Model')

    # Category와 SubCategory 컬럼 이름 정의
    leading_indices = list(range(model_idx))
    
    if len(leading_indices) >= 2:
        df_clean.rename(columns={df_clean.columns[leading_indices[-2]]: 'Category'}, inplace=True)
        df_clean.rename(columns={df_clean.columns[leading_indices[-1]]: 'SubCategory'}, inplace=True)
    
    # 판매량 컬럼 리스트 (Total 제외)
    month_cols = [col for col in df_clean.columns if col in MONTH_MAP]

    if not month_cols:
        st.warning(f"'{filepath}' 파일에서 월별 판매량 컬럼을 찾을 수 없습니다.")
        return pd.DataFrame()

    # 'Total' 행 및 모델명이 없는 행 제거
    df_clean = df_clean[~df_clean['Model'].astype(str).str.contains('total|Total', na=False, case=False)]
    df_clean = df_clean.dropna(subset=['Model'])

    # Wide 포맷을 Long 포맷으로 변환 (월별 데이터 통합)
    id_vars = ['Category', 'SubCategory', 'Model']
    
    # 실제로 존재하는 id_vars만 선택
    id_vars_present = [col for col in id_vars if col in df_clean.columns]
    
    df_long = pd.melt(
        df_clean,
        id_vars=id_vars_present,
        value_vars=month_cols,
        var_name='Month_Name',
        value_name='Sales_Volume'
    )

    # 데이터 정리
    df_long = df_long.dropna(subset=['Sales_Volume'])
    df_long['Sales_Volume'] = pd.to_numeric(df_long['Sales_Volume'], errors='coerce')
    df_long = df_long.dropna(subset=['Sales_Volume'])

    df_long['Year'] = year
    df_long['Month'] = df_long['Month_Name'].map(MONTH_MAP)
    
    # 날짜 컬럼 생성
    # 2025년 1월 데이터처럼 뒤가 비어있는 경우가 있으므로 오류 무시
    df_long['Date'] = pd.to_datetime(df_long[['Year', 'Month']].assign(day=1), errors='coerce')
    
    return df_long.dropna(subset=['Date'])


def combine_all_data():
    """모든 엑셀 파일을 로드하고 통합합니다."""
    all_data = []
    for filepath, year in ALL_FILES:
        df_temp = load_and_clean_data(filepath, year)
        if not df_temp.empty:
            all_data.append(df_temp)
    
    if not all_data:
        st.error("처리 가능한 데이터가 없어 예측을 진행할 수 없습니다. 파일을 확인해 주세요.")
        return pd.DataFrame()

    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined = df_combined.sort_values('Date').reset_index(drop=True)
    return df_combined

# --- 예측 함수 (단순 선형 트렌드 회귀 기반) ---

def perform_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """
    총 월별 판매량을 기반으로 2026년 수요를 예측합니다.
    (선형 회귀 모델 사용)
    """
    st.subheader("2026년 수요 예측 실행 중...")
    
    # 1. 월별 총 판매량 집계
    df_monthly = df.groupby('Date')['Sales_Volume'].sum().reset_index()
    df_monthly.columns = ['Date', 'Sales_Volume']
    
    # 2. 시계열 인덱스(Time Index) 생성
    # 2023년 1월 (첫 번째 데이터 포인트)을 0으로 설정
    first_date = df_monthly['Date'].min()
    df_monthly['Time_Index'] = (df_monthly['Date'] - first_date).dt.days // 30.4375 
    
    # 3. 간단한 선형 회귀 모델 학습 (scikit-learn 대신 numpy/pandas 기반으로 구현)
    X = df_monthly['Time_Index'].values.reshape(-1, 1)
    y = df_monthly['Sales_Volume'].values
    
    # 평균 및 기울기 계산 (OLS: Ordinary Least Squares)
    X_mean = X.mean()
    y_mean = y.mean()
    
    # 기울기 (m) = Sum((Xi - X_mean) * (yi - y_mean)) / Sum((Xi - X_mean)^2)
    # 절편 (b) = y_mean - m * X_mean
    numerator = ((X - X_mean) * (y - y_mean)).sum()
    denominator = ((X - X_mean)**2).sum()
    
    if denominator == 0:
        st.warning("데이터 포인트가 너무 적어 트렌드 분석을 할 수 없습니다. 평균값으로 예측합니다.")
        m = 0
    else:
        m = numerator / denominator
        
    b = y_mean - m * X_mean
    
    # 4. 2026년 예측 인덱스 생성
    last_index = df_monthly['Time_Index'].max()
    
    # 2026년 1월부터 12월까지의 Time Index 계산
    forecast_dates = pd.to_datetime(pd.date_range(start='2026-01-01', periods=12, freq='MS'))
    
    # 마지막 데이터 날짜를 기준으로 2026년 날짜들의 인덱스 계산
    forecast_indices = (forecast_dates - first_date).days // 30.4375

    # 5. 예측 값 계산
    forecast_sales = m * forecast_indices + b
    
    # 6. 예측 DataFrame 생성
    df_forecast = pd.DataFrame({
        'Date': forecast_dates,
        'Sales_Volume': forecast_sales.round(0).astype(int),
        'Type': 'Forecast',
        'Time_Index': forecast_indices
    })
    
    # Historical DataFrame 준비
    df_historical = df_monthly.copy()
    df_historical['Type'] = 'Historical'
    
    # 최종 통합 및 정리
    df_final = pd.concat([df_historical, df_forecast], ignore_index=True)
    df_final['Sales_Volume'] = df_final['Sales_Volume'].clip(lower=0) # 판매량이 음수가 되지 않도록 조정
    
    st.success("2026년 수요 예측 완료.")
    return df_final.sort_values('Date').reset_index(drop=True)

# --- Gemini API 분석 함수 ---

def generate_gemini_analysis(df_forecast: pd.DataFrame) -> str:
    """
    Gemini API를 호출하여 예측 결과에 대한 전문적인 분석을 요청합니다.
    """
    st.subheader("Gemini AI 분석 요청 중...")
    
    # 주요 예측 데이터 준비 (최근 과거 데이터 포함)
    analysis_data = df_forecast.copy()
    
    # 분석에 사용할 데이터 포인트만 추출 (최근 6개월 과거 + 12개월 예측)
    relevant_data = analysis_data.tail(18)
    
    # 텍스트 포맷으로 변환
    data_points = relevant_data.to_string(index=False)

    system_prompt = (
        "당신은 북미 자동차 시장의 최고 금융 분석가입니다. "
        "제공된 과거 및 예측 데이터를 바탕으로 2026년 북미 자동차 시장 수요에 대한 "
        "전문적이고 상세한 보고서를 작성해야 합니다. "
        "보고서에는 주요 트렌드, YoY(전년 대비) 성장률 예측, 수요를 주도할 핵심 요소에 대한 "
        "간결하고 심도 있는 분석이 포함되어야 합니다. 보고서는 한국어로 작성하고, "
        "데이터 출처는 언급하지 마십시오."
    )
    
    user_query = (
        "다음은 북미 자동차 시장의 월별 총 판매량 데이터입니다. "
        "'Historical'은 과거 데이터이며, 'Forecast'는 2026년 예측치입니다. "
        "이 데이터를 바탕으로 2026년 수요 예측 및 분석 보고서를 작성해 주세요.\n\n"
        f"데이터:\n{data_points}"
    )


    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={API_KEY}"
    
    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}], # 최신 정보 기반 분석을 위해 Google Search grounding 사용
    }

    headers = {'Content-Type': 'application/json'}
    
    # Exponential Backoff을 사용하여 API 호출
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status() # HTTP 오류 발생 시 예외 발생
            result = response.json()
            
            # 응답에서 텍스트 추출
            candidate = result.get('candidates', [{}])[0]
            text = candidate.get('content', {}).get('parts', [{}])[0].get('text', '')
            
            if text:
                st.success("Gemini AI 분석 완료.")
                return text
            else:
                st.error("Gemini 분석 결과가 비어 있습니다.")
                return "Gemini 분석을 로드하는 데 실패했습니다: 결과 텍스트 없음."

        except requests.exceptions.RequestException as e:
            st.warning(f"API 요청 실패 (시도 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                # 지수 백오프 대기: 2^attempt 초
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                st.error("최대 재시도 횟수를 초과했습니다. Gemini 분석을 로드하는 데 실패했습니다.")
                return "Gemini 분석을 로드하는 데 실패했습니다: 네트워크/API 오류."
        except Exception as e:
            st.error(f"예기치 않은 오류 발생: {e}")
            return "Gemini 분석을 로드하는 데 실패했습니다: 예기치 않은 오류."

# --- 엑셀 다운로드 함수 (이전 기능 유지) ---

def create_download_link(df: pd.DataFrame, filename: str, link_text: str):
    """
    DataFrame을 엑셀 파일로 변환하고 다운로드 링크를 생성합니다.
    """
    output = BytesIO()
    # openpyxl 엔진 사용
    with pd.ExcelWriter(output, engine='openpyxl') as writer: 
        df.to_excel(writer, index=False, sheet_name='Combined Sales Data')
    processed_data = output.getvalue()
    
    b64 = base64.b64encode(processed_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}" class="download-button">{link_text}</a>'
    return href

# --- Streamlit 메인 함수 ---

def main():
    st.set_page_config(layout="wide", page_title=ST_PAGE_TITLE)
    
    # Streamlit 스타일 및 헤더
    st.markdown(
        f"""
        <style>
        .download-button {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.75rem 1.25rem;
            border-radius: 0.5rem;
            color: #ffffff !important;
            background-color: #1e40af;
            text-decoration: none;
            font-weight: 600;
            border: none;
            transition: background-color 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        .download-button:hover {{
            background-color: #1d4ed8;
            color: #ffffff !important;
        }}
        .header-text {{
            font-size: 28px;
            font-weight: bold;
            color: #1f2937;
            margin-bottom: 20px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}
        .analysis-box {{
            border: 1px solid #3b82f6;
            border-radius: 0.5rem;
            padding: 15px;
            background-color: #eff6ff;
        }}
        </style>
        <div class="header-text">{ST_PAGE_TITLE}</div>
        """,
        unsafe_allow_html=True,
    )
    
    
    # 1. 데이터 로드 및 통합
    df_combined = combine_all_data()

    if df_combined.empty:
        st.error("통합된 데이터가 없어 대시보드를 표시할 수 없습니다.")
        return

    # 2. 데이터 미리보기 및 다운로드
    with st.expander("원본 데이터 통합 결과 및 다운로드", expanded=False):
        st.dataframe(df_combined.head(20), use_container_width=True, height=300)
        
        # 다운로드 링크 생성 (원래 기능 유지)
        excel_filename = "combined_retail_sales_long_format.xlsx"
        download_link = create_download_link(
            df_combined, 
            excel_filename, 
            f"통합된 월별 Long 포맷 데이터 ({excel_filename}) 다운로드"
        )
        st.markdown(download_link, unsafe_allow_html=True)
        st.info(f"총 {len(df_combined):,}개의 월별 모델 판매 기록이 통합되었습니다.")
    
    st.markdown("---")
    
    # 3. 예측 실행
    df_forecast_result = perform_forecasting(df_combined)
    
    # 4. 시각화
    try:
        import plotly.express as px
        
        st.subheader("북미 자동차 총 수요 예측 트렌드 (2023-2026)")
        
        # Plotly 차트 생성
        fig = px.line(
            df_forecast_result, 
            x='Date', 
            y='Sales_Volume', 
            color='Type', 
            title='월별 총 판매량: 과거 및 2026년 예측 (단위: 대)',
            labels={'Sales_Volume': '총 판매량', 'Date': '날짜', 'Type': '데이터 종류'},
            color_discrete_map={'Historical': '#1e40af', 'Forecast': '#ef4444'} # 파란색 (과거), 빨간색 (예측)
        )
        fig.update_layout(xaxis_title="날짜", yaxis_title="총 판매량 (대)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 2026년 예측 데이터 요약
        df_2026 = df_forecast_result[df_forecast_result['Type'] == 'Forecast']
        if not df_2026.empty:
            total_2026 = df_2026['Sales_Volume'].sum()
            avg_2026 = df_2026['Sales_Volume'].mean().round(0).astype(int)
            st.metric(label="2026년 예측 총 수요 (합계)", value=f"{total_2026:,} 대")
            st.caption(f"월평균 예측 판매량: {avg_2026:,} 대")
        
    except ImportError:
        st.warning("Plotly 모듈을 로드할 수 없습니다. 시각화 없이 테이블만 표시합니다.")
        st.dataframe(df_forecast_result, use_container_width=True)

    # 5. Gemini 분석 실행
    st.markdown("---")
    gemini_analysis_text = generate_gemini_analysis(df_forecast_result)
    
    st.subheader("Gemini AI 2026년 수요 분석 보고서")
    st.markdown(f'<div class="analysis-box">{gemini_analysis_text}</div>', unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()