# -*- coding: utf-8 -*-
"""
data_loader.py
- rag_data/ 폴더 전체에서 지원 확장자(.txt .md .pdf .docx .csv) 파일을 재귀적으로 읽어
  하나의 큰 문자열로 합칩니다.
- TXT/MD는 인코딩을 자동 감지(utf-8, utf-8-sig, cp949, euc-kr, iso-8859-1)합니다.

필요 패키지:
  pip install PyPDF2 docx2txt python-docx pandas
"""

# data_loader.py

import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from PyPDF2 import PdfReader

# --- 각 파일 타입별 로더 import ---
# PDF 로더 (pip install pypdf)
from langchain_community.document_loaders import PyPDFLoader
# DOCX 로더 (pip install python-docx docx2txt)
from langchain_community.document_loaders import Docx2txtLoader
# CSV 로더 (pip install pandas)
from langchain_community.document_loaders import CSVLoader
# 텍스트/마크다운 로더
from langchain_community.document_loaders import TextLoader

# 지원하는 파일 확장자와 로더 매핑
LOADER_MAPPING = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".csv": CSVLoader,
    # .txt, .md 등은 TextLoader로 처리
    ".txt": TextLoader,
    ".md": TextLoader,
}

def load_single_document(file_path: Path) -> List[Document]:
    """단일 파일을 로드하고 Document 리스트로 반환"""
    ext = file_path.suffix.lower()
    
    if ext in [".txt", ".md"]:
        # TextLoader는 인코딩 자동 감지를 시도 (euc-kr, cp949)
        try:
            loader = TextLoader(str(file_path), encoding="utf-8")
            return loader.load()
        except UnicodeDecodeError:
            try:
                # UTF-8 실패 시 CP949 시도
                loader = TextLoader(str(file_path), encoding="cp949")
                return loader.load()
            except Exception as e:
                print(f" [!] '{file_path.name}' 로드 실패 (CP949): {e}")
                return []
    
    elif ext in LOADER_MAPPING:
        # 기타 로더 (PyPDFLoader, Docx2txtLoader, CSVLoader)
        try:
            loader_class = LOADER_MAPPING[ext]
            loader = loader_class(str(file_path))
            return loader.load()
        except Exception as e:
            print(f" [!] '{file_path.name}' 로드 실패: {e}")
            return []
            
    else:
        # 지원하지 않는 확장자
        print(f" [!] '{file_path.name}' (은)는 지원하지 않는 파일 형식입니다: {ext}")
        return []

def load_all_from_data(data_dir: str | Path) -> List[Document]:
    """
    지정된 폴더(data_dir) 내의 모든 지원 파일을 Document 리스트로 로드합니다.
    
    """
    data_path = Path(data_dir)
    if not data_path.is_dir():
        raise FileNotFoundError(
            f"지식베이스 폴더를 찾을 수 없습니다: {data_path}\n"
            f"- 스크립트와 같은 위치에 'rag_data' 폴더를 생성하세요."
        )

    all_docs: List[Document] = []
    
    print(f"[KB] '{data_path}' 폴더에서 문서 로드를 시작합니다...")
    
    # data_path 내의 모든 파일/폴더 순회
    for file_path in data_path.rglob("*"):
        # 파일이 아니거나, .으로 시작하는 숨김 파일(예: .DS_Store)이면 건너뛰기
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        # 단일 파일 로드 시도
        docs = load_single_document(file_path)
        
        if docs:
            print(f"  [+] {file_path.name} 로드 완료 ({len(docs)}개 Document)")
            all_docs.extend(docs)

    if not all_docs:
        raise ValueError(
            f"'{data_path}' 폴더에 로드할 수 있는 문서(.txt, .pdf, .docx, .csv)가 없습니다."
        )

    print(f"[KB] 총 {len(all_docs)}개의 Document 로드 완료.")
    return all_docs