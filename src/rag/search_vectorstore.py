# FAISS 검색만 확인하는 파일

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 프로젝트 루트 기준 경로 계산
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vectorstore")

# FAISS 벡터 DB에서 query와 의미적으로 가장 유사한 문서 k개 검색
def search_emergency_guide(query: str, k: int = 3):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vectorstore.similarity_search(query, k=k)

    return docs


if __name__ == "__main__":
    query = "사람이 감전된 것 같고 의식이 없습니다"

    results = search_emergency_guide(query, k=3)

    print(f"\n🔍 질문: {query}")
    print("=" * 60)

    for i, doc in enumerate(results, 1):
        print(f"\n📄 결과 {i}")
        print(f"출처: {doc.metadata.get('source')}")
        print("-" * 40)
        print(doc.page_content)