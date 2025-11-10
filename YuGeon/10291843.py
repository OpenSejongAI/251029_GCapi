# -*- coding: utf-8 -*-
"""
동아리 챗봇 (Vertex AI + LangChain + FAISS, CLI 인증 기반, 안정판)

사전 준비:
1) gcloud 인증
   $ gcloud auth login
   $ gcloud config set project <YOUR_PROJECT_ID>
   (권장) ADC 쿼터 프로젝트 지정:
   $ gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>

2) 패키지 설치
   $ python -m pip install --upgrade google-cloud-aiplatform langchain-google-vertexai langchain
   $ python -m pip install --upgrade langchain-community langchain-text-splitters faiss-cpu

3) 지식베이스 파일 준비
   - 같은 폴더에 club_info.txt 를 두세요 (UTF-8 권장)
   - CP949/EUC-KR 등이어도 자동 감지합니다.

환경변수(선택):
  VERTEX_LOCATION       기본 us-central1
  VERTEX_MODEL          기본 gemini-2.5-pro
  VERTEX_EMBEDDING      기본 gemini-embedding-001
  LLM_TEMPERATURE       기본 0.4
  CLUB_INFO_PATH        기본 ./club_info.txt
  RETRIEVER_K           기본 5
  FAISS_DIR             기본 ./.faiss_club
"""

import os
import sys
import traceback
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


# ---------- 기본 설정 ----------
DEFAULT_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
KB_PATH = os.getenv("CLUB_INFO_PATH", "rag_data/club_info.txt")
VERTEX_MODEL = os.getenv("VERTEX_MODEL", "gemini-2.5-pro")
VERTEX_EMBEDDING = os.getenv("VERTEX_EMBEDDING", "gemini-embedding-001")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.4"))
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
FAISS_DIR = os.getenv("FAISS_DIR", "../.faiss_club")


# ---------- 유틸: 다중 인코딩 로더 ----------
def load_knowledge(path: str) -> str:
    """다양한 인코딩을 시도해 club_info 텍스트를 안전하게 읽는다."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"지식베이스 파일을 찾을 수 없습니다: {path}\n"
            f"- 같은 폴더에 'club_info.txt'를 두거나,\n"
            f"- 환경변수 CLUB_INFO_PATH 로 경로를 지정하세요."
        )
    candidates = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "iso-8859-1"]
    for enc in candidates:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            print(f"[KB] 파일 인코딩 감지: {enc}")
            return text
        except UnicodeDecodeError:
            continue
    # 최후 수단: 손상 문자 대체
    with open(path, "rb") as f:
        raw = f.read()
    print("[KB] 인코딩 감지 실패: utf-8(errors='replace')로 복구")
    return raw.decode("utf-8", errors="replace")


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
        kb_text: str,
        location: str = DEFAULT_LOCATION,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
):
    """
    텍스트 분할 → 임베딩 → (FAISS 저장/재사용) → Retriever → ChatVertexAI(Gemini 2.5 Pro)
    """
    # 1) Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = splitter.split_text(kb_text)
    if not texts:
        raise ValueError("지식베이스에서 텍스트 조각을 생성하지 못했습니다. club_info.txt 내용을 확인하세요.")
    if len(texts) < 3:
        print("[경고] 지식 조각이 매우 적습니다. club_info.txt 내용을 늘리는 것을 권장합니다.")

    # 2) Embeddings (Vertex AI)
    embeddings = VertexAIEmbeddings(model_name=VERTEX_EMBEDDING, location=location)

    # 3) Vector store (FAISS) — 저장/재사용
    faiss_path = Path(FAISS_DIR)
    faiss_path.mkdir(parents=True, exist_ok=True)
    faiss_index_file = faiss_path / "index.faiss"

    if faiss_index_file.exists():
        # 기존 인덱스 로드
        vectorstore = FAISS.load_local(
            str(faiss_path),
            embeddings,
            allow_dangerous_deserialization=True,  # FAISS 로드 시 필요
        )
        print(f"[FAISS] 기존 인덱스 로드: {faiss_path}")
    else:
        # 새 인덱스 생성 후 저장
        vectorstore = FAISS.from_texts(texts, embeddings)
        vectorstore.save_local(str(faiss_path))
        print(f"[FAISS] 인덱스 생성 및 저장: {faiss_path}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})

    # 4) LLM (Gemini on Vertex AI)
    llm = ChatVertexAI(
        model_name=VERTEX_MODEL,
        location=location,
        temperature=LLM_TEMPERATURE,
    )

    # 5) RAG Chain (LCEL 기반, langchain.chains 미사용)
    prompt = ChatPromptTemplate.from_template(
        "너는 동아리 안내 챗봇이다. 주어진 컨텍스트로만 답하라. "
        "모르면 '자료에 없습니다'라고 간단히 말하라.\n\n"
        "컨텍스트:\n{context}\n\n질문:\n{question}"
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    qa_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
    )
    return qa_chain


# ---------- 대화 루프 ----------
def interactive_loop(qa_chain):
    print("\n[동아리 챗봇] 준비 완료! (종료: /quit)")
    while True:
        try:
            q = input("\n질문 > ").strip()
            if not q:
                continue
            if q.lower() in {"/q", "/quit", "quit", "exit"}:
                print("종료합니다. 👋")
                break

            # LCEL 체인은 입력 문자열 q를 그대로 받아 'question'에 전달
            result = qa_chain.invoke(q)
            answer = getattr(result, "content", str(result))

            print(f"\n답변\n----\n{answer}")

        except KeyboardInterrupt:
            print("\n강제 종료되었습니다. 👋")
            break
        except Exception as e:
            print("\n오류가 발생했습니다:")
            print(e)
            traceback.print_exc(limit=1)


# ---------- 엔트리 포인트 ----------
def main():
    # 0) Vertex AI 초기화 (ADC)
    init_vertex(location=DEFAULT_LOCATION)

    # 1) 지식베이스 로드
    kb_text = load_knowledge(KB_PATH)

    # 2) RAG 체인 구성
    qa_chain = build_rag_chain(kb_text, location=DEFAULT_LOCATION)

    # 3) 대화 루프
    interactive_loop(qa_chain)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[프로그램 오류]")
        print(exc)
        sys.exit(1)