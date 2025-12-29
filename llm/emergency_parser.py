import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dotenv import load_dotenv
from openai import OpenAI
from config import LLM_API_KEY

# 환경 변수 로드
load_dotenv()

if not LLM_API_KEY:
    raise RuntimeError("GEMMA_API_KEY not found")

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL_NAME = "models/gemma-3-4b-it"


# 프롬프트 정의

SYSTEM_PROMPT = """
너는 의료 진단을 수행하지 않는다.
너의 역할은 응급 상황에 대한 사용자 입력을 분석하여,
필요한 의료 자원과 중증도를 구조화된 데이터로 분류하는 것이다.

다음 규칙을 반드시 지켜라:
- 진단이나 치료 방법을 제시하지 말 것
- 병원 추천을 하지 말 것
- 반드시 JSON 형식으로만 출력할 것
- 지정된 enum 값 외의 값은 사용하지 말 것
- 정보가 불충분하면 UNKNOWN 또는 false로 설정할 것
"""

USER_PROMPT_TEMPLATE = """
다음은 환자의 응급 상황에 대한 설명이다.

[환자 설명]
"{user_input}"

위 설명을 분석하여 아래 JSON 스키마에 맞게 출력하라.

출력 JSON 스키마:
{{
  "severity": "LOW | MEDIUM | HIGH",
  "suspected_condition": "CARDIAC | RESPIRATORY | NEURO | TRAUMA | BURN | POISON | PEDIATRIC | UNKNOWN",
  "required_resources": {{
    "need_icu": true | false,
    "need_ventilator": true | false,
    "need_ct": true | false,
    "need_mri": true | false
  }},
  "notes": "환자 상태 요약 (1문장)",
  "confidence": 0.0 ~ 1.0
}}
"""

# JSON 추출 함수

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

# Gemma 파싱 함수

def parse_emergency_text(user_input: str) -> dict:
    full_prompt = SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE.format(
        user_input=user_input
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": full_prompt}
        ],
        temperature=0.2
    )

    raw_text = response.choices[0].message.content.strip()
    json_text = extract_json(raw_text)

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON 파싱 실패:\n{json_text}") from e

    return parsed

# 테스트 실행

if __name__ == "__main__":
    test_input = "어떤 남자가 흉부에 칼을 찔려서 쓰러져있습니다."

    result = parse_emergency_text(test_input)

    print("=== LLM 파싱 결과 ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    print("\n=== 변수 확인 ===")
    print("severity:", result["severity"])
    print("condition:", result["suspected_condition"])
    print("need_icu:", result["required_resources"]["need_icu"])
    print("need_ventilator:", result["required_resources"]["need_ventilator"])
    print("need_ct:", result["required_resources"]["need_ct"])
    print("need_mri:", result["required_resources"]["need_mri"])