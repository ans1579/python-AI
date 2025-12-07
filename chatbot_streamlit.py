# chatbot_streamlit.py

import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# 환경 변수 로드 (.env 파일에서 GEMINI_API_KEY 로드)
load_dotenv()

# [ 환경 변수 및 Gemini API 설정 ]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    # 키가 없으면 에러 표시 후 종료
    st.error("❌ GEMINI_API_KEY 환경 변수가 설정되어 있지 않습니다.")
    st.stop()

# Gemini API 설정
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"
SYSTEM_INSTRUCTION = (
    "당신은 고객의 문의에 친절하고 정확하게 답변하는 전문 상담 AI 봇입니다. "
    "사용자의 질문에 한국어로만 응답하며, 전문적인 금융/차량 상담 용어를 쉽게 설명해 줍니다."
)


# [ 챗봇 응답 생성 함수 ]
def generate_gemini_response(prompt: str):
    """API를 호출하여 응답을 생성합니다."""
    
    # 세션 상태에서 Chat 객체를 가져오거나 새로 만듭니다.
    # st.session_state에 'chat_session'이 없으면 새로 생성합니다.
    if 'chat_session' not in st.session_state:
        st.session_state.chat_session = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_INSTRUCTION
        ).start_chat(history=[])

    # Chat Session을 사용하여 메시지 전송
    response = st.session_state.chat_session.send_message(prompt)
    return response.text


# [ Streamlit UI 구현 ]
st.set_page_config(page_title="Gemini Streamlit 챗봇", layout="centered")
st.title("실시간 상담 AI 봇")
st.caption("AI 봇과의 실시간 대화를 시작해보세요.")

# 1. 채팅 기록 초기화
# st.session_state에 'messages' 리스트가 없으면 초기화합니다.
if "messages" not in st.session_state:
    # 챗봇을 시작할 때 초기 메시지 제공
    st.session_state.messages = [{"role": "model", "content": "안녕하세요! 저는 전문 상담 AI 봇입니다. 어떤 것을 도와드릴까요?"}]

# 2. 기존 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 사용자 입력 처리
if prompt := st.chat_input("여기에 질문을 입력하세요..."):
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