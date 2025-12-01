from fastapi import FastAPI, Request
from pydantic import BaseModel

from chatbot import answer, get_qa_chain  # get_qa_chain 가져오기

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

app = FastAPI(title="Club Chatbot API (RAG + Vertex AI)")


@app.on_event("startup")
async def startup_event():
    """
    서버가 시작될 때 RAG 체인을 미리 초기화해서
    첫 요청에서 타임아웃이 나지 않도록 만든다.
    """
    try:
        print("[Startup] RAG 체인 초기화 시작")
        get_qa_chain()
        print("[Startup] RAG 체인 초기화 완료")
    except Exception as e:
        print("[Startup] RAG 초기화 중 오류:", e)


def generate_answer(user_message: str) -> str:
    try:
        return answer(user_message)
    except Exception as e:
        return f"챗봇 처리 중 오류가 발생했어. 잠시 후 다시 시도해줘!\n(내부 오류: {e})"


@app.get("/")
def health_check():
    return {"status": "ok", "message": "clubbot api running with RAG"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    reply = generate_answer(req.message)
    return ChatResponse(reply=reply)


@app.post("/kakao")
async def kakao_webhook(request: Request):
    body = await request.json()
    user_msg = body.get("userRequest", {}).get("utterance", "")
    reply = generate_answer(user_msg)

    return {
        "version": "2.0",
        "template": {
            "outputs": [
                {
                    "simpleText": {
                        "text": reply
                    }
                }
            ]
        }
    }
