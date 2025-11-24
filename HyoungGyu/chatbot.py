# -*- coding: utf-8 -*-
"""
동아리 챗봇 (Vertex AI + LangChain + FAISS, CLI 인증 기반, 안정판; 멀티파일 지원)

사전 준비:
1) gcloud 인증
   $ gcloud auth login
   $ gcloud config set project <YOUR_PROJECT_ID>
   (권장) ADC 쿼터 프로젝트 지정:
   $ gcloud auth application-default set-quota-project <YOUR_PROJECT_ID>

2) 패키지 설치
   $ python -m pip install --upgrade google-cloud-aiplatform langchain-google-vertexai langchain
   $ python -m pip install --upgrade langchain-community langchain-text-splitters faiss-cpu
   $ python -m pip install --upgrade PyPDF2 docx2txt python-docx pandas

3) 지식 소스 준비
   - 프로젝트의 rag_data/ 폴더에 .txt .md .pdf .docx .csv 파일을 넣으세요.
   - TXT/MD는 CP949/EUC-KR도 자동 감지합니다(UTF-8 권장).

환경변수(선택):
  VERTEX_LOCATION       기본 us-central1
  VERTEX_MODEL          기본 gemini-2.5-pro
  VERTEX_EMBEDDING      기본 gemini-embedding-001
  LLM_TEMPERATURE       기본 0.4
  RETRIEVER_K           기본 5
  FAISS_DIR             기본 ./.faiss_club
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
import sys
import traceback
import time
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

# --- [ 여기를 수정합니다 ] ---
# 스크립트 파일(.py)이 있는 폴더의 절대 경로
SCRIPT_DIR = Path(__file__).resolve().parent
# 스크립트 폴더 기준의 .faiss_club 경로
DEFAULT_FAISS_DIR = SCRIPT_DIR / ".faiss_club"
FAISS_DIR = os.getenv("FAISS_DIR", str(DEFAULT_FAISS_DIR))
# ------------------------


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
# ---------- RAG 체인 ----------
def build_rag_chain(
        documents: list,  # <-- [2/3] 'kb_text: str' 에서 'documents: list'로 수정
        location: str = DEFAULT_LOCATION,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
):
    """
    (문서 리스트) 분할 → 임베딩 → (FAISS 저장/재사용) ...
    """
    # 1) Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # --- [2/3] 'split_text(kb_text)'를 'split_documents(documents)'로 수정 ---
    texts = splitter.split_documents(documents)
    # ------------------------------------------------------------------

    if not texts:
        raise ValueError("지식 텍스트 분할 실패. rag_data 내 파일 내용을 확인하세요.")
    if len(texts) < 3:
        print("[경고] 지식 조각이 매우 적습니다. 자료를 더 넣는 것을 권장합니다.")

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
            allow_dangerous_deserialization=True,
        )
        print(f"[FAISS] 기존 인덱스 로드: {faiss_path}")
    else:
        # 새 인덱스 생성 후 저장
        # --- [2/3] 'FAISS.from_texts'를 'FAISS.from_documents'로 수정 ---
        vectorstore = FAISS.from_documents(texts, embeddings)
        # --------------------------------------------------------------
        vectorstore.save_local(str(faiss_path))
        print(f"[FAISS] 인덱스 생성 및 저장: {faiss_path}")

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
    
    # ... (이하 함수 내용은 동일) ...
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


# ---------- 대화 루프 ----------
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

            # --- [3/3] 응답 시간 측정 시작 ---
            start_time = time.time()

            # LCEL 체인은 입력 문자열 q를 그대로 받아 'question'에 전달
            result = qa_chain.invoke(q)
            answer = getattr(result, "content", str(result))

            end_time = time.time()
            duration = end_time - start_time
            # --- [3/3] 응답 시간 측정 종료 ---

            print(f"\n답변\n----\n{answer}")

            # --- [3/3] 응답 시간 출력 (소수점 2자리) ---
            print(f"\n(응답 시간: {duration:.2f}초)")
            # ------------------------------------

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

    # --- [ 여기를 수정합니다 ] ---
    # 스크립트 파일(.py) 기준 경로 계산
    SCRIPT_DIR = Path(__file__).resolve().parent
    ROOT_DIR = SCRIPT_DIR.parent
    # 프로젝트 루트의 rag_data 사용
    KB_DIR = ROOT_DIR / "rag_data"

    # 1) 지식 소스 로드 (수정된 경로 사용)
    all_docs = load_all_from_data(KB_DIR)
    # ------------------------

    # 2) RAG 체인 구성 (이전 단계에서 수정한 all_docs 리스트 전달)
    qa_chain = build_rag_chain(all_docs, location=DEFAULT_LOCATION)

    # 3) 대화 루프
    interactive_loop(qa_chain)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\n[프로그램 오류]")
        print(exc)
        sys.exit(1)
