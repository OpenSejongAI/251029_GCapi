# -*- coding: utf-8 -*-
"""
동아리 챗봇 (Vertex AI + LangChain + FAISS, CLI 인증 기반, 안정판; 멀티파일 지원)
웹 서버(FastAPI, 카카오 스킬 서버) / CLI 양쪽에서 공통으로 사용할 수 있도록
get_qa_chain(), answer() 를 제공한다.
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import traceback
import time
from functools import lru_cache
from typing import Optional
from pathlib import Path

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

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FAISS_DIR = SCRIPT_DIR / ".faiss_club"
FAISS_DIR = os.getenv("FAISS_DIR", str(DEFAULT_FAISS_DIR))


# ---------- Vertex 초기화 ----------
def init_vertex(project: Optional[str] = None, location: str = DEFAULT_LOCATION) -> str:
    """
    gcloud CLI(ADC) 기반 인증 자동 사용.
    프로젝트는 google.auth.default()에서 감지하거나 인자로 지정.
    """
    creds, detected_project = google.auth.default()
    project_id = project or detected_project
    if not project_id:
        raise RuntimeError(
            "프로젝트 ID를 찾을 수 없습니다. gcloud 설정을 확인하세요.\n"
            "예) gcloud config set project <YOUR_PROJECT_ID>"
        )
    aiplatform.init(project=project_id, location=location)
    print(f"[Vertex AI] 프로젝트: {project_id}, 리전: {location}")
    return project_id


# ---------- RAG 체인 ----------

def build_rag_chain(
    documents: list,
    location: str = DEFAULT_LOCATION,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
):
    """
    (문서 리스트) 분할 → 임베딩 → (FAISS 저장/재사용) → retriever + LLM 체인 생성
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = splitter.split_documents(documents)

    if not texts:
        raise ValueError("지식 텍스트 분할 실패. rag_data 내 파일 내용을 확인하세요.")
    if len(texts) < 3:
        print("[경고] 지식 조각이 매우 적습니다. 자료를 더 넣는 것을 권장합니다.")

    # 1) Embeddings (Vertex AI)
    embeddings = VertexAIEmbeddings(
        model_name=VERTEX_EMBEDDING,
        location=location,
    )

    # 2) Vector store (FAISS) — 저장/재사용
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
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(str(faiss_path))
        print(f"[FAISS] 인덱스 생성 및 저장: {faiss_path}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 3) LLM + 프롬프트
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


# ---------- 전역 RAG 체인 (웹/CLI 공통) ----------

@lru_cache(maxsize=1)
def get_qa_chain():
    """
    Vertex 초기화 + 지식 로드 + RAG 체인 생성.
    서버에서 여러 번 호출돼도 실제로는 한 번만 초기화되도록 lru_cache 사용.
    """
    # 0) Vertex AI 초기화 (ADC)
    init_vertex(location=DEFAULT_LOCATION)

    # 1) 지식 소스 로드
    kb_dir = SCRIPT_DIR / "rag_data"
    all_docs = load_all_from_data(kb_dir)

    # 2) RAG 체인 구성
    qa_chain = build_rag_chain(all_docs, location=DEFAULT_LOCATION)
    print("[RAG] qa_chain 초기화 완료")
    return qa_chain


def answer(question: str) -> str:
    """
    외부(FastAPI, 카카오, Flask 등)에서 사용할 메인 엔트리.
    질문 문자열을 받아 답변 문자열만 반환.
    """
    qa_chain = get_qa_chain()

    start_time = time.time()
    result = qa_chain.invoke(question)
    end_time = time.time()

    # LangChain 결과 객체에서 content만 꺼내기
    answer_text = getattr(result, "content", str(result))
    print(f"[RAG] 질문 처리 완료 (응답 시간: {end_time - start_time:.2f}초)")
    return answer_text


# ---------- 대화 루프 (CLI용) ----------

def interactive_loop():
    """
    터미널에서 직접 챗봇을 사용하고 싶을 때 사용하는 CLI 루프.
    """
    qa_chain = get_qa_chain()
    print("\n[동아리 챗봇] 준비 완료! (종료: /quit)\n")
    while True:
        try:
            q = input("질문 > ").strip()
            if not q:
                continue
            if q.lower() in {"/q", "/quit", "quit", "exit"}:
                print("종료합니다. 👋")
                break

            start_time = time.time()
            result = qa_chain.invoke(q)
            answer_text = getattr(result, "content", str(result))
            end_time = time.time()
            duration = end_time - start_time

            print(f"\n답변\n----\n{answer_text}")
            print(f"\n(응답 시간: {duration:.2f}초)\n")

        except KeyboardInterrupt:
            print("\n강제 종료되었습니다. 👋")
            break
        except Exception as e:
            print("\n오류가 발생했습니다:")
            print(e)
            traceback.print_exc(limit=1)


# ---------- 엔트리 포인트 ----------

def main():
    # CLI로 실행할 때만 사용
    try:
        interactive_loop()
    except Exception as exc:
        print("\n[프로그램 오류]")
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
