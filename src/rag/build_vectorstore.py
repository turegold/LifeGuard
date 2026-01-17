import os
import re
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
GUIDE_DIR = os.path.join(PROJECT_ROOT, "data", "guide")
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, "data", "vectorstore")

CATEGORY_RE = re.compile(r"\[CATEGORY\]\s*\n\s*([A-Z_]+)", re.MULTILINE)


def extract_category(text: str) -> str:
    m = CATEGORY_RE.search(text or "")
    return m.group(1).strip() if m else "UNKNOWN"


def build_vectorstore():
    documents = []

    if not os.path.exists(GUIDE_DIR):
        raise RuntimeError(f"GUIDE_DIR not found: {GUIDE_DIR}")

    for root, _, files in os.walk(GUIDE_DIR):
        for file in files:
            if not file.endswith(".txt"):
                continue

            path = os.path.join(root, file)
            loader = TextLoader(path, encoding="utf-8")
            loaded_docs = loader.load()

            for doc in loaded_docs:
                category = extract_category(doc.page_content)
                doc.metadata = doc.metadata or {}
                doc.metadata["category"] = category
                doc.metadata["source_file"] = file
                doc.metadata["source_path"] = os.path.relpath(path, GUIDE_DIR)
                documents.append(doc)

    print(f"📄 로드된 문서 수: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata.setdefault("category", "UNKNOWN")
        c.metadata.setdefault("source_file", "UNKNOWN")
        c.metadata.setdefault("source_path", "UNKNOWN")

    print(f"✂️ 생성된 청크 수: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    if os.path.exists(VECTOR_DB_DIR):
        shutil.rmtree(VECTOR_DB_DIR)
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    vectorstore.save_local(VECTOR_DB_DIR)
    print("✅ HuggingFace + FAISS VectorStore 생성 완료")


if __name__ == "__main__":
    build_vectorstore()