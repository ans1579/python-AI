# -*- coding: utf-8 -*-
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_BASE = "http://localhost:8090/5core"
st.set_page_config(page_title="판매실적", layout="wide", initial_sidebar_state="collapsed")

# 헤더 및 사이드바 숨김 CSS
st.markdown("""
<style>
    section[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)
def load_data(dealer_id=None):
    try:
        if dealer_id is None:
            url = f"{API_BASE}/api/sales/vehicle-sales/me"
        else:
            url = f"{API_BASE}/api/sales/vehicle-sales/me?dealerId={dealer_id}"
        
        resp = requests.get(url, timeout=10)
        
        if resp.status_code == 200:
            data = resp.json()
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

# 쿼리 파라미터 우선 → 수동 입력
dealer_id = st.query_params.get("dealerId", [None])[0]
if dealer_id is None or dealer_id == "":
    dealer_id = st.number_input("딜러 ID", value=3, min_value=1, step=1, key="dealer_input")
else:
    dealer_id = int(dealer_id)

# 데이터 로드
df = load_data(dealer_id if dealer_id != "None" else None)

# 1. 원 그래프
if not df.empty:
    fig = px.pie(df.nlargest(10, "salesCount"), 
                 values="salesCount", names="vehicleName", 
                 hole=0.4, height=400)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(showlegend=False, margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # 2. KPI: 총 판매대수, 취급차종
    total_sales = int(df["salesCount"].sum())
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("총 판매대수", f"{total_sales:,}")
    with col2:
        st.metric("취급 차종", len(df))
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # 3. 상세 내역
    st.markdown("### 상세 내역")
    display_df = df[["vehicleName", "salesCount"]].sort_values("salesCount", ascending=False)
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # 4. 딜러ID 표시
    st.markdown("---")
    st.caption(f"딜러ID: {dealer_id}")
    
else:
    # 빈 데이터일 때도 순서 유지
    fig = px.pie(values=[1], names=["데이터없음"], hole=0.4, height=400)
    fig.update_traces(textposition='inside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.metric("총 판매대수", "0")
    with col2:
        st.metric("취급 차종", "0")
    
    st.markdown("### 상세 내역")
    st.dataframe(pd.DataFrame(columns=["vehicleName", "salesCount"]), hide_index=True)
    
    st.markdown("---")
    st.caption(f"딜러ID: {dealer_id}")
    st.warning("판매 실적 데이터가 없습니다.")
