# chatbot_streamlit_manager.py

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# 환경 변수 로드 (.env 파일에서 GEMINI_API_KEY 로드)
load_dotenv()

# [ 환경 변수 및 Gemini API 설정 ]
# API Key가 .env 파일에 설정되어 있어야 합니다.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # 키가 없으면 에러 표시 후 종료
    st.error("❌ GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
    st.stop()

# Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"
# ⭐ 관리자 업무에 맞게 SYSTEM_INSTRUCTION 변경
SYSTEM_INSTRUCTION = (
    "당신은 현대자동차 관리자 및 실무자를 위한 **전문 업무 지원 AI 봇**입니다. "
    "사용자의 질문에 한국어로만 응답하며, 내부 규정, 생산 계획, 품질 관리, "
    "자재 수급 현황 등 **현대자동차 내부 업무 지식**을 기반으로 정확하고 "
    "신속하게 답변합니다. 답변은 항상 객관적이고 명료하게 제시해야 합니다."
)


# [ 챗봇 응답 생성 함수 ]
def generate_gemini_response(prompt: str):
    """API를 호출하여 응답을 생성합니다."""
    
    # 세션 상태에서 Chat 객체를 가져오거나 새로 만듭니다.
    if 'chat_session' not in st.session_state:
        # 시스템 지침을 적용하여 새로운 채팅 세션 시작
        st.session_state.chat_session = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        ).start_chat(history=[])

    # Chat Session을 사용하여 메시지 전송
    response = st.session_state.chat_session.send_message(prompt)
    return response.text


# [ Streamlit UI 구현 ]
st.set_page_config(page_title="관리자용 챗봇", layout="centered")
st.title("관리자 업무 지원 AI 봇")
st.caption("생산, 품질, 자재 및 내부 규정에 대한 업무 지원을 시작합니다.")

# 1. 채팅 기록 초기화
if "messages" not in st.session_state:
    # ⭐ 관리자에게 적합한 초기 메시지 제공
    st.session_state.messages = [
        {"role": "model", "content": "안녕하세요, 저는 관리자님을 위한 **업무 지원 AI**입니다. **금주의 생산 목표 달성률**이나 **최신 품질 관리 지침** 등 궁금한 점을 문의해 주세요."}
    ]

# 2. 기존 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 사용자 입력 처리
if prompt := st.chat_input("생산 현황, 규정 검색 등 질문을 입력하세요..."):
    # 사용자 메시지를 기록 및 화면에 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. AI 응답 생성 및 표시
    with st.spinner("AI가 응답을 생성 중입니다..."):
        full_response = generate_gemini_response(prompt)
        
    with st.chat_message("model"):
        # AI 응답을 화면에 표시
        st.markdown(full_response)
        
    # AI 메시지를 기록에 추가
    st.session_state.messages.append({"role": "model", "content": full_response})