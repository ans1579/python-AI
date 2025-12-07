# -*- coding: utf-8 -*-
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_BASE = "http://localhost:8090/5core"

st.set_page_config(
    page_title="월별실적",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# URL에서 dealerId 받기 (없으면 3)
dealer_id = st.query_params.get("dealerId", ["3"])[0]

@st.cache_data(ttl=30)
def load_monthly_data(dealer_id_str, year=2025):
    try:
        dealer_id_int = int(dealer_id_str) if dealer_id_str.isdigit() else 3
        url = f"{API_BASE}/api/sales/monthly-sales?dealerId={dealer_id_int}&year={year}"
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data)
        df["salesCount"] = pd.to_numeric(df["salesCount"]).fillna(0)
        return df.sort_values("month")
    except Exception:
        return pd.DataFrame({
            "month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
            "salesCount": [0]*12,
        })

df = load_monthly_data(dealer_id)

if not df.empty:
    total_count = int(df["salesCount"].sum())
    total_amount = 45_000_000 * total_count
    avg_monthly = int(df["salesCount"].mean())
    latest_month = df.iloc[-1]["month"]
    latest_count = int(df["salesCount"].iloc[-1])

    # 그래프
    fig = px.line(
        df, x="month", y="salesCount",
        markers=True, line_shape="spline",
        labels={"salesCount": ""},
        color_discrete_sequence=["#1e40af"],
    )
    fig.update_layout(
        height=260,
        margin=dict(t=0, b=10, l=20, r=20),
        showlegend=False,
        font_size=10,
    )
    fig.update_traces(line_width=3, marker_size=5)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("총매출", f"₩{total_amount:,}")
    with col2:
        st.metric("총대수", f"{total_count:,}대")