# app.py
# Vertex RAG (club_rag_bot.py) 위에 얹는 Flask 서버

from flask import Flask, request, jsonify
from werkzeug.exceptions import HTTPException

# 네가 방금 수정한 RAG 코드 파일 이름에 맞게 import
# 예: 파일 이름이 club_rag_bot.py 라면 아래처럼
from chatbot import generate_answer, init_qa_chain

app = Flask(__name__)

# 서버 시작 시 한 번 RAG 체인 미리 초기화(선택)
# 초기 로딩 시간을 줄이고 싶으면 사용
try:
    init_qa_chain()
    print("[Flask] RAG 체인 미리 초기화 완료")
except Exception as e:
    print("[Flask] RAG 초기화 중 오류 (처음 요청 시 다시 시도 예정):", e)


@app.route("/ping", methods=["GET"])
def ping():
    """헬스체크용 엔드포인트"""
    return "pong", 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    일반 JSON용 엔드포인트 (테스트 / 다른 클라이언트용)
    요청 예:
      { "message": "동아리 소개해줘" }
    응답 예:
      { "answer": "..." }
    """
    try:
        data = request.get_json() or {}
        user_msg = (data.get("message") or "").strip()

        if not user_msg:
            return jsonify({"error": "message 필드가 비어 있습니다."}), 400

        answer = generate_answer(user_msg)
        return jsonify({"answer": answer})

    except Exception as e:
        print("[/chat] 오류:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/kakao", methods=["POST"])
def kakao():
    """
    카카오 i 오픈빌더 Webhook용 엔드포인트

    카카오에서 들어오는 JSON 형식(대표적인 예):
    {
      "userRequest": {
        "utterance": "동아리 지원하려면 어떻게 해요?"
      },
      ...
    }

    카카오가 기대하는 응답 형식:
    {
      "version": "2.0",
      "template": {
        "outputs": [
          {
            "simpleText": {
              "text": "모델의 답변"
            }
          }
        ]
      }
    }
    """
    try:
        body = request.get_json() or {}
        user_req = body.get("userRequest", {})
        user_msg = (user_req.get("utterance") or "").strip()

        if not user_msg:
            # 카카오 쪽에 에러를 친절하게 알려줄 수도 있고, 그냥 안내 멘트로 처리해도 됨
            answer = "질문을 받지 못했습니다. 다시 입력해 주세요."
        else:
            answer = generate_answer(user_msg)

        kakao_response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": answer
                        }
                    }
                ]
            }
        }
        return jsonify(kakao_response)

    except Exception as e:
        print("[/kakao] 오류:", e)
        # 카카오도 JSON을 기대하므로 에러일 때도 JSON으로 응답
        fail_msg = f"서버에서 오류가 발생했습니다: {e}"
        kakao_response = {
            "version": "2.0",
            "template": {
                "outputs": [
                    {
                        "simpleText": {
                            "text": fail_msg
                        }
                    }
                ]
            }
        }
        return jsonify(kakao_response), 200  # 카카오는 보통 200 응답만 기대


# 전역 에러 핸들러(선택)
@app.errorhandler(Exception)
def handle_exception(e):
    """예상 못한 예외 처리용(주로 디버깅용)"""
    if isinstance(e, HTTPException):
        return e
    print("[Flask 글로벌 예외]", e)
    return jsonify({"error": "서버 내부 오류가 발생했습니다."}), 500


if __name__ == "__main__":
    # 개발용 실행
    # host="0.0.0.0" 으로 두면 ngrok 같은 걸로 외부에서 터널링하기 좋음
    app.run(host="0.0.0.0", port=8000, debug=True)


