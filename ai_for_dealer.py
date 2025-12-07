# app.py

import os
import json
import streamlit as st
import requests
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import argparse


# [ 5core 연동 기본 설정 ]

load_dotenv(override=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
st.write("현재 사용 중인 GEMINI_API_KEY =", GEMINI_API_KEY[:8] if GEMINI_API_KEY else "없음")

FIVECORE_BASE_URL = os.environ.get("FIVECORE_BASE_URL", "http://localhost:8090/5core")

# Counseling 상세 조회 경로
COUNSELING_DETAIL_PATH = "/api/counseling/{counselingId}"

# 상담 정보 기반 차량 후보 조회 경로
VEHICLE_CANDIDATE_PATH = "/api/vehicles/ai-candidates"

# AI 추천 결과 저장 경로
AI_RECOMMEND_SAVE_PATH = "/api/counseling/{counselingId}/ai-recommendations"

# 최종 추천 결과 저장 경로
FINAL_CHOICE_SAVE_PATH = "/api/counseling/{counselingId}/final-recommendations"


# [ 환경변수, 제미나이 설정 ]

if not GEMINI_API_KEY:
    st.error("API KEY가 설정되어 있지 않습니다.")
    st.stop()

genai.configure(api_key = GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

system_instruction = """
당신은 현대 자동차 딜러입니다.
사용자는 차량 구매 상담을 신청한 고객이며, 5core의 상담(Counseling) 정보를 기반으로 상담을 진행합니다.

Counseling 엔티티에 포함되어 있는 정보:
    vehicleId, vehicleName, purchasePurpose, otherInput, purchasePeriod, vehicleType, engineType, customer, dealer 등

당신의 역할:
    1. 상담 정보와 차량 후보 리스트를 보고, 고객에게 적합한 차량 후보들을 여러 대 추천합니다.
    2. 각 차량에 대해 추천하는 이유를 작성합니다.
    3. 항상 JSON 형식으로만 응답합니다.

"""

model = genai.GenerativeModel(
    model_name = MODEL_NAME,
    system_instruction = system_instruction
)


# [ 5core 연동 함수 ]

def build_url(path_template: str, **path_vars) -> str:
    path = path_template.format(**path_vars)
    return f"{FIVECORE_BASE_URL}{path}"

# 상담 상세 정보를 조회
def get_counseling(counseling_id: int):
    url = build_url(COUNSELING_DETAIL_PATH, counselingId = counseling_id)
    
    try:
        res = requests.get(url, timeout = 5)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Counseling 조회 실패: {e}")
        return None

# 상담의 추가 요구 사항을 조회
def extract_need(counseling_json: dict) -> dict:
    if not counseling_json:
        return {}
    
    hopearea = counseling_json.get("hopearea") or {}

    return {
        "vehicleId" : counseling_json.get("vehicleId"),
        "vehicleName" : counseling_json.get("vehicleName"),
        "purchasePurpose" : counseling_json.get("purchasePurpose"),
        "otherInput" : counseling_json.get("otherInput"),
        "vehicleType" : counseling_json.get("vehicleType"),
        "engineType" : counseling_json.get("engineType")
    }

# counseling 요구사항을 기반으로 후보 차량들을 조회
def get_vehicles(need_info: dict) -> dict:

    url = build_url(VEHICLE_CANDIDATE_PATH)

    data = {
        "vehicleType" : need_info.get("vehicleType"),
        "engineType" : need_info.get("engineType"),
        "purchasePurpose" : need_info.get("purchasePurpose")
    }

    try: 
        res = requests.get(url, params = data, timeout = 5)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.warning(f"차량 후보 조회 실패, 더미 데이터 사용: {e}")
        return [
            {
                "modelCode": "avante",
                "name": "아반떼",
                "vehicleType": "세단",
                "engineType": "가솔린",
                "basePrice": 20348000,
                "fuelType": "휘발유"
            },
            {
                "modelCode": "tucson",
                "name": "투싼",
                "vehicleType": "SUV",
                "engineType": "디젤",
                "basePrice": 32527000,
                "fuelType": "디젤"
            },
            {
                "modelCode": "ioniq5",
                "name": "아이오닉 5",
                "vehicleType": "SUV",
                "engineType": "전기",
                "basePrice": 54790000,
                "fuelType": "전기"
            }
        ]

# 차량 이미지 URL 생성
def get_vehicle_image_url(recommends: dict) -> str:
    # 추천 차량 정보의 fileName을 읽어서 URL 생성
    # /5core/images/{fileName} 형태의 URL을 만들어 줌
    file_name = recommends.get("fileName")
    return f"http://localhost:8090/5core/images/{file_name}"


# 차량 이미지와 가격을 불러오지 못하는 문제 해결
# 추천된 차량의 modelCode로 fileName과 finalPrice를 불러와서 저장
def get_recommended_vehicles_info(recommends: list, candidates: list) -> None:
    # modelCode를 소문자로 변환 (기호는 상관없음)
    small_code = {str(c.get("modelCode")).lower(): c for c in candidates}

    for r in recommends:
        code = str(r.get("modelCode")).lower()
        src = small_code.get(code)
        if not src:
            continue

        # 이미지 파일명 저장
        if not r.get("fileName"):
            r["fileName"] = src.get("fileName")
        
        # 가격 저장
        if not r.get("finalPrice"):
            r["finalPrice"] = src.get("finalPrice")



# AI가 뽑은 차량 후보 리스트를 5core에 저장 - 딜러가 확인하는 용도
def save_ai_vehicle_list(counseling_id: int, result: dict) -> bool:

    url = build_url(AI_RECOMMEND_SAVE_PATH, counselingId = counseling_id)
    payload = {
        "saved" : datetime.now().isoformat(),
        "result" : result
    }
    
    try:
        res = requests.post(url, json = payload, timeout = 5)
        res.raise_for_status()
        return True
    except Exception as e:
        st.error(f"차량 후보 리스트 저장 실패: {e}")
        return False

# 딜러가 선택한 최종 추천 차량 1대를 5core에 저장
def save_final_choice(counseling_id: int, choice: dict) -> bool:

    url = build_url(FINAL_CHOICE_SAVE_PATH, counselingId = counseling_id)
    payload = {
        "saved" : datetime.now().isoformat(),
        "vehicle" : choice
    }

    try:
        res = requests.post(url, json = payload, timeout = 5)
        res.raise_for_status()
        return True
    except Exception as e:
        st.error(f"최종 추천 차량 저장 실패: {e}")
        return False
    

# [ Gemini 호출 ]

# 상담 + 요구사항 + 후보 차량 리스트를 Gemini에게 넘겨서 차량 추천받기
def ask_gemini(counseling_json: dict, need_info: dict, candidates: list, max_count: int = 4) -> dict:

    counseling_text = json.dumps(counseling_json, ensure_ascii = False, indent = 2)
    need_text = json.dumps(need_info, ensure_ascii = False, indent = 2)
    candidates_text = json.dumps(candidates, ensure_ascii = False, indent = 2)

    # Prompt로 Gemini에게 추천 방식 및 형식을 지정해줌
    prompt = f"""
다음은 5core 시스템에 저장된 상담(Counseling) 정보입니다.

[Couneling 전체 JSON]
{counseling_text}

아래는 Counseling에서 추출한 고객 요구사항입니다.

[고객 요구사항(need)]
{need_text}

아래는 이 고객에게 적합할 수 있는 후보 차량 리스트입니다.

[후보 차량 리스트]
{candidates_text}

역할:
    - 너는 이 정보들을 바탕으로 딜러에게 보여줄 추천 후보 차량을 골라야한다.
    - 딜러가 이 리스트를 보고 최종 1대를 선택하여 고객에게 추천할 것이다.
    - 추천 후보 차량은 최대 {max_count}대까지 선택해라.

반드시 아래 JSON 형식으로만 응답할 것. 추가 설명 문장은 금지함.

예시형식:
    {{
        "추천": [
            {{
                "modelCode": "avante",
                "name": "아반떼",
                "vehicleType": "세단",
                "fuelType": "휘발유",
                "reason": "고객은 출퇴근 목적의 세단을 선호하고, 가격적으로 저렴하여 좋습니다."
            }},
            {{
                "modelCode": "tucson",
                "name": "투싼",
                "vehicleType": "SUV",
                "fuelType": "경유",
                "reason": "고객은 패밀리카를 선호하고, SUV를 선호하여 추천합니다."
            }},
            {{
                "modelCode": "ioniq5",
                "name": "아이오닉 5",
                "vehicleType": "SUV",
                "fuelType": "전기",
                "reason": "고객은 전기차를 선호하고, SUV를 선호하여 추천합니다."
            }}
        ]
    }}
"""
    response = model.generate_content(prompt)
    res = response.text

    try:
        jText = res.strip()
        # json형식 문자열 앞뒤의 `을 제거
        if jText.startswith("```"):
            jText = jText.strip("`")
            # 문자열을 줄단위로 분리
            lines = jText.splitlines()
            if lines and "json" in lines[0].lower():
                jText = "\n".join(lines[1:])
            
        data = json.loads(jText)
        return data
    except Exception as e:
        st.error(f"Gemini의 응답 JSON 파싱 실패: {e}")
        st.code(res)
        return {"추천": []}

# [ Streamlit 설정 ]
st.set_page_config(page_title = "5core AI Recommends", layout = "wide")
st.subheader("5core AI Recommends for Dealer")

# streamlit 전체적인 html설정 (css 작성)
st.markdown(
    """
    <style>
    /* 전체적인 색 설정 */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }

    .block-container {
        max-width: 1600px;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    /* 카드 컨테이너 공통 스타일 */
    div[data-testid="stContainer"] {
        padding: 6px 8px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
        margin-bottom: 8px;
        background-color: #ffffff;
    }

    /* 카드 이미지 */
    div[data-testid="stContainer"] img {
        max-height: 120px;
        object-fit: contain;
    }

    /* 컬럼 좌우 여백 */
    div[data-testid="column"] {
        padding-left: 4px !important;
        padding-right: 4px !important;
    }

    div.stButton > button[kind="secondary"] {
        padding: 16px 0 !important;   /* 버튼 높이 */
        font-size: 16px !important;   /* 글자 크기 */
        font-weight: 600 !important;  /* 글자 굵기 */
        border-radius: 999px !important; /* 둥글게 */
    }
    

    </style>
    """,
    unsafe_allow_html=True,
)


# 실행 인자 + URL 에서 상담 ID 가져오기
# 커맨드라인 인자 파싱 (예: --counselingId=1)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--counselingId")
args, _ = parser.parse_known_args()
default_counseling_id = args.counselingId  # 문자열 또는 None

# URL 쿼리파라미터에서 먼저 시도 (?counselingId=1)
params = st.query_params

counseling_id = None

c_id = params.get("counselingId", None)
if isinstance(c_id, list):
    c_id = c_id[0]

if c_id:
    try:
        counseling_id = int(c_id)
    except ValueError:
        st.error(f"URL의 counselingId가 숫자가 아닙니다: {c_id}")
        st.stop()

# URL에 없으면 실행 인자(--counselingId)를 사용
elif default_counseling_id is not None:
    try:
        counseling_id = int(default_counseling_id)
    except ValueError:
        st.error(f"실행 인자의 counselingId가 잘못되었습니다: {default_counseling_id}")
        st.stop()

# 둘 다 없으면 에러
if counseling_id is None:
    st.error("URL 또는 실행 인자로 counselingId가 필요합니다. 예: ?counselingId=1")
    st.stop()

counseling_json = get_counseling(counseling_id)


# 세션 상태
if "counseling_json" not in st.session_state:
    st.session_state["counseling_json"] = None
if "need_info" not in st.session_state:
    st.session_state["need_info"] = None
if "candidates" not in st.session_state:
    st.session_state["candidates"] = None
if "recommend_result" not in st.session_state:
    st.session_state["recommend_result"] = None


# 상담 정보 불러오기
# - URL의 상담ID를 통해 자동으로 불러오기
if st.session_state["counseling_json"] is None:
    counseling_json = get_counseling(counseling_id)
    if counseling_json:
        st.session_state["counseling_json"] = counseling_json
        st.session_state["need_info"] = extract_need(counseling_json)
        # st.success(f"상담 정보를 불러왔습니다. (상담ID = {counseling_id})")
    else:
        st.error("상담 정보를 불러오지 못했습니다.")
        st.stop()

counseling_json = st.session_state["counseling_json"]
need_info = st.session_state["need_info"]

############################
# JSON 정보는 주석처리 해둠
# st.subheader("상담 정보")
# st.json(counseling_json)

# st.subheader("고객 요구사항")
# st.json(need_info)
############################

# 후보 차량 자동 조회
if st.session_state["candidates"] is None:
    with st.spinner("후보 차량을 조회하는 중입니다..."):
        candidates = get_vehicles(need_info or {})
    st.session_state["candidates"] = candidates
    st.session_state["recommend_result"] = None
    # st.success(f"후보 차량 {len(candidates)}대를 불러왔습니다.")

candidates = st.session_state["candidates"]

############################
# JSON 정보는 주석처리 해둠
# if candidates:
#     st.subheader("후보 차량 리스트")
#     st.json(candidates)
# else:
#     st.warning("조건에 맞는 차량이 없습니다.")
#     st.stop()
############################

# AI 추천 자동 생성
MAX_COUNT = 4

if st.session_state["recommend_result"] is None:
    with st.spinner("AI가 추천을 진행중입니다..."):
        result = ask_gemini(
            counseling_json = counseling_json,
            need_info = need_info,
            candidates = candidates,
            max_count = MAX_COUNT
        )
    
    # 추천 리스트에서 이미지 경로와 가격 저장
    # - 이미지와 가격 정보를 불러오지 못하는 문제 해결
    recommends = result.get("추천")
    get_recommended_vehicles_info(recommends, candidates)

    st.session_state["recommend_result"] = result
    # st.success("AI 추천이 끝났습니다.")


# --- AI 추천 결과 표시 ---
recommend_result = st.session_state["recommend_result"]

if recommend_result:
    # JSON은 주석 처리했음
    # st.json(recommend_result)

    recommends = recommend_result.get("추천") or []

    # 전체후보 중 4대를 추천
    recommends = recommends[:4]
    
    if recommends:
        st.markdown("AI 추천 결과")

        # 처음 들어왔을 때 첫번째가 선택되어있음 기본 선택값 = 0번
        # 선택된 인덱스가 현재 추천 개수보다 크다면 0으로 초기화
        # 이유가 선택되어있는 것 (딜러의 최종선택 x)
        if (
            "selected_reco_idx" not in st.session_state
            or st.session_state["selected_reco_idx"] >= len(recommends)
        ):
            st.session_state["selected_reco_idx"] = 0

        # 한 줄에 2개씩 배치
        cards_per_row = 2

        for row_start in range(0, len(recommends), cards_per_row):
            cols = st.columns(cards_per_row)
            row_items = recommends[row_start:row_start + cards_per_row]

            for col_offset, (col, r) in enumerate(zip(cols, row_items)):
                idx = row_start + col_offset

                name = r.get("name", "")
                vtype = r.get("vehicleType", "")
                fuel = r.get("fuelType", "")
                price = r.get("finalPrice")
                img_url = get_vehicle_image_url(r)

                with col:
                    with st.container(border=True):   # Streamlit 내장 Container
                        st.image(img_url)

                        st.markdown(
                            f"<p style='font-size:17px; font-weight:700; margin:4px 0 2px 0;'>{name}</p>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<p style='margin:0; font-size:15px;'>차종: {vtype}</p>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<p style='margin:0; font-size:15px;'>연료: {fuel}</p>",
                            unsafe_allow_html=True,
                        )
                        if price is not None:
                            st.markdown(
                                f"<p style='margin:0 0 12px 0; font-size:15px;'>가격: {price:,} 원</p>",
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                "<p style='margin:0 0 6px 0; font-size:15px;'>가격: -</p>",
                                unsafe_allow_html=True,
                            )

                        if st.button(f"왜 {name}일까?", key=f"select_{idx}", use_container_width=True):
                            st.session_state["selected_reco_idx"] = idx
                

        # 아래에 선택된 차량 이유 보여주기
        selected_idx = st.session_state["selected_reco_idx"]
        selected = recommends[selected_idx]

st.markdown("<hr style='margin:8px 0;' />", unsafe_allow_html=True)


# 이유
st.markdown("### 차량을 추천하는 이유")
st.markdown(
    f"""
**{selected.get('name', '')}** 를 추천합니다:

> {selected.get('reason', '이유가 없습니다.')}
"""
)

# 이유 아래에 여백 조금 주고
st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

# 버튼을 가운데 정렬하고 싶으면 열 3개 만들어서 가운데 열에만 버튼 배치
btn_left, btn_center, btn_right = st.columns([1, 2, 1])
with btn_center:
    if st.button("선택한 차량을 최종 추천", key="final_choice", use_container_width=True):
        ok = save_final_choice(counseling_id, selected)
        if ok:
            st.success("최종 추천 차량을 5core에 저장했습니다.")






