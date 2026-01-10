import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from openai import OpenAI
from config import LLM_API_KEY

# =========================
# 환경 변수 로드
# =========================
load_dotenv()

if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY not found")

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL_NAME = "models/gemma-3-4b-it"


# =========================
# SYSTEM PROMPT (Gemma용)
# ⚠️ system role 사용 불가 → user prompt에 포함
# =========================
SYSTEM_PROMPT = """
너는 의료 진단이나 치료 판단을 수행하지 않는다.
너의 역할은 이미 계산된 병원 랭킹 결과를 바탕으로,
"왜 이 병원이 추천되었는지"를 사용자에게 이해하기 쉽게 설명하는 것이다.

반드시 아래 규칙을 지켜라:
- 의료적 진단, 치료 방법, 처방을 제시하지 말 것
- 병원 순위를 변경하거나 새로운 병원을 추천하지 말 것
- 제공된 데이터만 사용하여 설명할 것
- 응급 상황을 고려하여 간결하고 명확하게 설명할 것
- 반드시 JSON 형식으로만 출력할 것
"""


# =========================
# USER PROMPT TEMPLATE
# =========================
USER_PROMPT_TEMPLATE = """
다음은 응급 환자 정보와 병원 추천 결과이다.

[환자 정보]
{patient_info}

[병원 추천 결과 (Top-K)]
{ranked_hospitals}

위 정보를 바탕으로,
각 병원이 왜 이 순위로 추천되었는지를 설명하라.

출력 JSON 스키마는 다음과 같다:

{{
  "summary": "전체 추천 결과 요약 (2~3문장)",
  "hospital_explanations": [
    {{
      "rank": 1,
      "hospital_name": "병원명",
      "explanation": "이 병원이 해당 순위로 추천된 이유 (1~2문장)"
    }}
  ]
}}
"""


# =========================
# JSON 추출 유틸
# =========================
def extract_json(text: str) -> str:
    text = text.strip()

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                return part

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        return text[start:end + 1]

    raise ValueError("JSON 영역을 찾을 수 없습니다.")


# =========================
# 병원 랭킹 설명 함수 (MAIN)
# =========================
def explain_hospital_ranking(
    ranked_results: list,
    patient_info: dict
) -> dict:
    """
    ranked_results: recommend_hospitals 결과
    patient_info: LLM emergency_parser 결과
    """

    if not ranked_results:
        return {
            "summary": "현재 조건에 맞는 병원 추천 결과가 없습니다.",
            "hospital_explanations": []
        }

    # LLM 입력용 정리
    ranked_hospitals_for_prompt = []
    for i, r in enumerate(ranked_results, start=1):
        ranked_hospitals_for_prompt.append({
            "rank": i,
            "hospital_name": r.get("hospital_name"),
            "accept_prob": round(r.get("accept_prob", 0), 3),
            "features": {
                k: v for k, v in r.items()
                if k not in ("hospital_name", "hospital_phone", "hospital_id")
            }
        })

    full_prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + USER_PROMPT_TEMPLATE.format(
            patient_info=json.dumps(patient_info, ensure_ascii=False, indent=2),
            ranked_hospitals=json.dumps(ranked_hospitals_for_prompt, ensure_ascii=False, indent=2)
        )
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.3
    )

    raw_text = response.choices[0].message.content.strip()
    json_text = extract_json(raw_text)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패:\n{json_text}") from e

    return parsed


# =========================
# 단독 실행 테스트
# =========================
if __name__ == "__main__":
    dummy_patient = {
        "severity": "HIGH",
        "suspected_condition": "TRAUMA",
        "required_resources": {
            "need_icu": True,
            "need_ventilator": True,
            "need_ct": True,
            "need_mri": False
        },
        "confidence": 0.82
    }

    dummy_results = [
        {
            "hospital_name": "경희대학교병원",
            "hospital_id": "A1100001",
            "hospital_phone": "02-958-8114",
            "accept_prob": 0.816,
            "filter_level": 0,
            "er_bed_ratio": 0.72,
            "icu_bed_ratio": 0.66,
            "distance_km": 4.8
        },
        {
            "hospital_name": "국립중앙의료원",
            "hospital_id": "A1100052",
            "hospital_phone": "02-2276-2114",
            "accept_prob": 0.518,
            "filter_level": 1,
            "er_bed_ratio": 0.55,
            "icu_bed_ratio": 0.41,
            "distance_km": 6.1
        }
    ]

    result = explain_hospital_ranking(dummy_results, dummy_patient)
    print(json.dumps(result, ensure_ascii=False, indent=2))