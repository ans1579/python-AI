# -*- coding: utf-8 -*-
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_BASE = "http://localhost:8090/5core"

st.set_page_config(page_title="판매실적", layout="wide", initial_sidebar_state="collapsed")

# 📊 판매 실적 헤더 완전 제거!

# memberId
member_id = st.query_params.get("memberId", ["45"])[0]
member_id = st.sidebar.number_input("memberId", value=int(member_id) if member_id else 45, min_value=1)

@st.cache_data(ttl=30)
def load_data(member_id):
    try:
        url = f"{API_BASE}/api/sales/vehicle-sales/me?memberId={member_id}"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame(data)
    except:
        return pd.DataFrame()

df = load_data(member_id)

if not df.empty:
    total_sales = int(df["salesCount"].sum())
    
    # 파이 차트 (최상단)
    fig = px.pie(
        df, values="salesCount", names="vehicleName",
        hole=0.4, title=None  # 제목 제거
    )
    fig.update_layout(
        height=260, 
        margin=dict(t=0, b=0, l=0, r=0), 
        showlegend=False
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='label+percent', 
        textfont_size=11
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # KPI 수평 배치 (판매대수 | 차종수)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("판매대수", total_sales)
    with col2:
        st.metric("차종수", len(df))
        
else:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("판매대수", 0)
    with col2:
        st.metric("차종수", 0)
