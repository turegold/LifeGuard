import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# =====================
# 경로 설정
# =====================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
GUIDE_DIR = os.path.join(PROJECT_ROOT, "data", "guide")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vectorstore")

def build_vectorstore():
    documents = []

    # 1️⃣ 가이드 문서 로드
    for file in os.listdir(GUIDE_DIR):
        if file.endswith(".txt"):
            path = os.path.join(GUIDE_DIR, file)
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())

    print(f"📄 로드된 문서 수: {len(documents)}")

    # 2️⃣ 문서 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    print(f"✂️ 생성된 청크 수: {len(chunks)}")

    # 3️⃣ HuggingFace 임베딩
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4️⃣ FAISS 벡터스토어 생성
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_DIR)

    print("✅ HuggingFace + FAISS VectorStore 생성 완료")

if __name__ == "__main__":
    build_vectorstore()