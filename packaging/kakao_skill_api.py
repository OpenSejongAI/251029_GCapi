# -*- coding: utf-8 -*-
"""
Flask 기반 Kakao 챗봇 스킬 서버

카카오 오픈빌더에서 스킬을 호출하면 이 서버의 /skill 엔드포인트로
POST 요청을 보내고, 서버는 Vertex AI + RAG 기반 답변을 Kakao 응답 포맷으로 반환한다.

실행:
    $ export FLASK_APP=kakao_skill_api:app
    $ flask run --host=0.0.0.0 --port=5000

테스트:
    curl -X POST http://localhost:5000/skill -H "Content-Type: application/json" -d '{"userRequest":{"utterance":"안녕"}}'
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request

from chatbot import DEFAULT_LOCATION, build_rag_chain, init_vertex
from data_loader import load_all_from_data

app = Flask(__name__)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# QA 체인은 최초 요청 시 한 번만 생성하고 재사용한다.
_qa_chain = None


def build_qa_chain_once():
    """Vertex 초기화 + RAG 체인 구성"""
    global _qa_chain
    if _qa_chain is not None:
        return _qa_chain

    init_vertex(location=DEFAULT_LOCATION)

    kb_dir = Path(__file__).resolve().parent.parent / "rag_data"
    documents = load_all_from_data(kb_dir)
    _qa_chain = build_rag_chain(documents, location=DEFAULT_LOCATION)
    logger.info("[KakaoSkill] QA chain initialized")
    return _qa_chain


def kakao_simple_text(text: str) -> Dict[str, Any]:
    """카카오 스킬 응답 포맷(simpleText) 생성"""
    # 카카오 제한은 1000자이므로 초과 시 잘라낸다.
    trimmed = text.strip()[:1000] or "죄송합니다. 답변을 생성하지 못했습니다."
    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {"simpleText": {"text": trimmed}},
            ]
        },
    }


@app.route("/healthz", methods=["GET"])
def healthcheck():
    """헬스 체크 엔드포인트"""
    try:
        build_qa_chain_once()
        status = "ok"
    except Exception as exc:  # pragma: no cover - 단순 헬스체크
        logger.exception("Healthcheck failed: %s", exc)
        status = "error"
    return jsonify({"status": status})


@app.route("/skill", methods=["POST"])
def skill():
    """카카오 챗봇 스킬 진입점"""
    payload = request.get_json(silent=True)
    if not payload:
        logger.warning("Empty payload from Kakao")
        return jsonify(kakao_simple_text("요청 데이터를 읽지 못했습니다.")), 400

    utterance = (
        payload.get("userRequest", {})
        .get("utterance", "")
        .strip()
    )
    if not utterance:
        logger.info("No utterance provided: %s", json.dumps(payload, ensure_ascii=False)[:200])
        return jsonify(kakao_simple_text("질문 내용을 찾을 수 없습니다.")), 200

    try:
        qa_chain = build_qa_chain_once()
        result = qa_chain.invoke(utterance)
        answer = getattr(result, "content", str(result))
    except Exception as exc:
        logger.exception("Failed to generate answer: %s", exc)
        return jsonify(kakao_simple_text("일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")), 500

    return jsonify(kakao_simple_text(answer)), 200


if __name__ == "__main__":
    # 개발 환경에서 직접 실행할 수 있도록 run() 추가
    app.run(host="0.0.0.0", port=5000)
