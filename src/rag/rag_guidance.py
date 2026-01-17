import json
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

from config import LLM_API_KEY

if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY not found")

client = OpenAI(
    api_key=LLM_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL_NAME = "models/gemma-3-4b-it"


# 경로
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vectorstore")


# SYSTEM PROMPT
SYSTEM_PROMPT = """
너는 의료 진단이나 치료를 하지 않는다.
아래에 제공된 문서 내용만 근거로 응급 행동 가이드를 작성한다.

규칙:
- 문서에 없는 내용은 절대 추가하지 말 것
- 치료법, 약물, 처방, 진단 금지
- 현재 상황에서 실제로 발생 가능한 행동만 포함할 것
- 다른 유형의 응급 상황에서만 의미 있는 행동은 제외할 것
- 행동 지침만 간결하게 정리
- 반드시 JSON 형식으로만 출력

중요:
- do_not_do 항목에는 현재 상황에서 실제로 사용자가 실수할 가능성이 있는 행동만 포함할 것
- 참고 문헌에 있더라도 현재 상황과 무관하면 절대 포함하지 말 것
"""


# USER PROMPT TEMPLATE
USER_PROMPT_TEMPLATE = """
[상황 설명]
{query}

[참고 문서]
{documents}

위 문서를 참고하여,
현재 상황에 맞는 응급 행동 가이드를 작성하라.
현재 상황과 다른 유형의 응급 상황은 절대 포함하지 말 것.


출력 JSON 형식:

{{
  "situation_summary": "상황 요약 (1문장)",
  "immediate_actions": [
    "즉시 해야 할 행동 1",
    "즉시 해야 할 행동 2"
  ],
  "do_not_do": [
    "절대 하면 안 되는 행동 1",
    "절대 하면 안 되는 행동 2"
  ]
}}
"""

# JSON 추출 유틸
def extract_json(text: str) -> dict:
    text = text.strip()

    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                return json.loads(part)

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])

    raise ValueError("JSON을 찾을 수 없습니다.")


# RAG + Gemma 메인 함수
def generate_emergency_guidance(
    query: str,
    condition: str,
    top_k: int = 5
) -> dict:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(query, k=top_k * 5)

    if condition:
        docs = [d for d in docs if d.metadata.get("category") == condition]

    docs = docs[:top_k]

    if not docs:
        return {
            "situation_summary": "현재 상황에 맞는 응급 가이드를 찾을 수 없습니다.",
            "immediate_actions": [
                "즉시 119에 신고",
                "환자의 의식과 호흡 상태를 확인"
            ],
            "do_not_do": [
                "임의로 판단하여 조치",
                "음식이나 물 제공"
            ]
        }

    documents_text = "\n\n".join(
        f"[문서 {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    full_prompt = (
        SYSTEM_PROMPT
        + "\n\n"
        + USER_PROMPT_TEMPLATE.format(
            query=query,
            documents=documents_text
        )
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2
    )

    return extract_json(response.choices[0].message.content)

# 단독 실행 테스트
if __name__ == "__main__":
    result = generate_emergency_guidance(
        query="어떤 남자가 흉부에 칼을 찔려서 쓰려져있습니다.",
        condition="TRAUMA"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))