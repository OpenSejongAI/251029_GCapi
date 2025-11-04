# -*- coding: utf-8 -*-
"""
data_loader.py
- rag_data/ 폴더 전체에서 지원 확장자(.txt .md .pdf .docx .csv) 파일을 재귀적으로 읽어
  하나의 큰 문자열로 합칩니다.
- TXT/MD는 인코딩을 자동 감지(utf-8, utf-8-sig, cp949, euc-kr, iso-8859-1)합니다.

필요 패키지:
  pip install PyPDF2 docx2txt python-docx pandas
"""

from pathlib import Path
import pandas as pd
import docx2txt
from PyPDF2 import PdfReader

SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".docx", ".csv"}


def _read_txt_any_encoding(path: Path) -> str:
    candidates = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "iso-8859-1"]
    for enc in candidates:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            print(f"[KB] 텍스트 인코딩 감지: {enc} ({path})")
            return text
        except UnicodeDecodeError:
            continue
    with open(path, "rb") as f:
        raw = f.read()
    print(f"[KB] 인코딩 감지 실패: utf-8(errors='replace')로 복구 ({path})")
    return raw.decode("utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    # docx2txt가 내부적으로 python-docx를 활용
    return docx2txt.process(str(path)) or ""


def _read_csv(path: Path) -> str:
    # 기본 시도 → 실패 시 cp949로 재시도
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="cp949")
    return df.to_string(index=False)


def _read_any_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return _read_txt_any_encoding(path)
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext == ".csv":
        return _read_csv(path)
    return ""


def load_all_from_data(data_dir: str = "rag_data") -> str:
    """
    data_dir 폴더를 재귀 탐색하여 지원 확장자의 파일을 모두 읽고,
    파일 경계마다 헤더(=== 파일명 ===)를 붙여 하나의 큰 문자열로 반환.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"{data_dir} 폴더가 없습니다. 생성 후 파일을 넣어주세요.")

    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            files.append(p)
    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"{data_dir} 폴더에 지원 확장자({', '.join(sorted(SUPPORTED_EXTS))}) 파일이 없습니다."
        )

    print("[KB] 로딩 대상 파일들:")
    for f in files:
        print("  -", f)

    chunks = []
    for f in files:
        try:
            text = _read_any_file(f)
            chunks.append(f"\n\n=== {f.name} ===\n{text}")
        except Exception as e:
            print(f"[경고] 파일 로딩 실패: {f} ({e})")

    big_text = "\n".join(chunks)
    print(f"[KB] 전체 텍스트 길이: {len(big_text):,} chars")
    return big_text