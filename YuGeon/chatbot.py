# -*- coding: utf-8 -*-
"""
동아리 챗봇 (Vertex AI + LangChain + FAISS, Flask API 기반)

환경변수(선택):
  VERTEX_LOCATION       기본 us-central1
  VERTEX_MODEL          기본 gemini-2.5-pro
  VERTEX_EMBEDDING      기본 gemini-embedding-001
  LLM_TEMPERATURE       기본 0.4
  RETRIEVER_K           기본 5
  FAISS_DIR             기본 ./.faiss_club
  RAG_DATA_PATH         기본 ../rag_data
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
from typing import Optional
from pathlib import Path

# Flask
from flask import Flask, request, jsonify

# Vertex AI & LangChain
from google.cloud import aiplatform
import google.auth
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# 외부 로더
from data_loader import load_all_from_data

# ---------- 기본 설정 ----------
DEFAULT_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
VERTEX_EMBEDDING = os.getenv("VERTEX_EMBEDDING", "gemini-embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
FAISS_DIR = os.getenv("FAISS_DIR", "./.faiss_club") # Docker 컨테이너 내 경로
RAG_DATA_PATH = os.getenv("RAG_DATA_PATH", "../rag_data") # Docker 컨테이너 내 경로


# ---------- Vertex 초기화 ----------
def init_vertex(project: Optional[str] = None, location: str = DEFAULT_LOCATION) -> str:
    creds, detected_project = google.auth.default()
    project_id = project or detected_project
    if not project_id:
        raise RuntimeError("프로젝트 ID를 찾을 수 없습니다. gcloud나 GOOGLE_APPLICATION_CREDENTIALS 설정을 확인하세요.")
    aiplatform.init(project=project_id, location=location)
    print(f"[Vertex AI] 프로젝트: {project_id}, 리전: {location}")
    return project_id


# ---------- RAG 체인 ----------
def build_rag_chain(
        rag_data_path: str,
        location: str = DEFAULT_LOCATION,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
):
    embeddings = VertexAIEmbeddings(model_name=VERTEX_EMBEDDING, location=location)
    faiss_path = Path(FAISS_DIR)
    faiss_path.mkdir(parents=True, exist_ok=True)
    faiss_index_file = faiss_path / "index.faiss"

    if faiss_index_file.exists():
        vectorstore = FAISS.load_local(
            str(faiss_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        print(f"[FAISS] 기존 인덱스 로드: {faiss_path}")
    else:
        print(f"[FAISS] 인덱스 없음. '{rag_data_path}'에서 데이터 로드 시작...")
        kb_text = load_all_from_data(rag_data_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = splitter.split_text(kb_text)
        if not texts:
            raise ValueError("지식 텍스트 분할 실패. rag_data 내 파일 내용을 확인하세요.")
        
        print("[FAISS] 임베딩 및 인덱스 생성 중...")
        vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(str(faiss_path))
        print(f"[FAISS] 인덱스 생성 및 저장 완료: {faiss_path}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    llm = ChatVertexAI(
        model_name=VERTEX_MODEL,
        location=location,
        temperature=LLM_TEMPERATURE,
    )
    prompt = ChatPromptTemplate.from_template(
        "너는 동아리 안내 챗봇이다. 주어진 컨텍스트로만 답하라. "
        "모르면 '자료에 없습니다'라고 간단히 말하라.\n\n"
        "컨텍스트:\n{context}\n\n질문:\n{question}"
    )

    def format_docs(docs):
        return "\n\n".join(getattr(d, "page_content", "") or "" for d in docs)

    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
    )
    return qa_chain

# ---------- Flask 앱 설정 ----------
app = Flask(__name__)

# 서버 시작 시 한 번만 모델과 RAG 체인을 로드합니다.
print("[Flask] 서버 초기화 시작...")
try:
    init_vertex(location=DEFAULT_LOCATION)
    qa_chain = build_rag_chain(RAG_DATA_PATH, location=DEFAULT_LOCATION)
    print("[Flask] 서버 준비 완료!")
except Exception as e:
    print(f"[Flask] 초기화 실패: {e}", file=sys.stderr)
    qa_chain = None # 초기화 실패 시 qa_chain을 None으로 설정

@app.route("/")
def index():
    return "챗봇 API 서버가 실행 중입니다. /ask 엔드포인트에 POST 요청을 보내세요.", 200

@app.route("/ask", methods=["POST"])
def ask():
    if not qa_chain:
        return jsonify({"error": "서버가 초기화되지 않았습니다. 로그를 확인하세요."}), 500

    req_data = request.get_json()
    if not req_data or "question" not in req_data:
        return jsonify({"error": "요청 형식이 잘못되었습니다. 'question' 필드가 필요합니다."}), 400

    question = req_data["question"]
    if not question:
        return jsonify({"error": "'question' 필드는 비워둘 수 없습니다."}), 400

    try:
        result = qa_chain.invoke(question)
        answer = getattr(result, "content", str(result))
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        return jsonify({"error": "답변을 생성하는 중 오류가 발생했습니다."}), 500

# Gunicorn과 같은 프로덕션 WSGI 서버에서 이 파일을 직접 실행하지 않으므로,
# if __name__ == "__main__": app.run() 블록은 로컬 테스트용으로만 사용되며,
# 프로덕션 배포 시에는 필요하지 않습니다.
