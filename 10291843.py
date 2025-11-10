import os
# (삭제) from langchain_google_genai import ...
# (추가) 'vertexai' 라이브러리를 임포트합니다.
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# (삭제) .env 파일 및 API 키 관련 코드가 모두 필요 없어집니다.
# ----------------------------------------------------
# load_dotenv()
# api_key = os.getenv("GOOGLE_API_KEY")
# if not api_key:
#     raise ValueError("...")
# ----------------------------------------------------
# 1단계에서 설정한 'GOOGLE_APPLICATION_CREDENTIALS' 환경 변수를
# 라이브러리가 자동으로 인식하여 인증합니다.


# 1. 지식 베이스(텍스트 파일) 로드
with open('club_info.txt', 'r', encoding='utf-8') as f:
    club_info = f.read()

# 2. 텍스트를 작은 조각(Chunk)으로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(club_info)

# ... (import 부분은 동일) ...

# 3. Google 'VertexAI' 임베딩 모델 로드
print("Vertex AI 임베딩 모델을 로드합니다...")
embeddings = VertexAIEmbeddings(
    model_name="textembedding-gecko@003",
    location="asia-northeast3"  # <-- (중요) 리전을 'asia-northeast3' (서울)로 지정
)
vectorstore = FAISS.from_texts(texts, embeddings)
print("임베딩 및 벡터 DB 생성 완료.")

# 4. Google 'VertexAI' Gemini 모델 로드
print("Vertex AI LLM 모델을 로드합니다...")
llm = ChatVertexAI(
    model_name="gemini-1.0-pro",
    location="asia-northeast3",  # <-- (중요) LLM에도 동일한 리전 지정
    temperature=0.7,
    convert_system_message_to_human=True
)
print("LLM 로드 완료.")

# ... (이하 RAG 체인 생성 및 실행 코드는 동일) ...

# 5. RAG 체인 생성 (최신 LangChain 1.0 방식)
print("RAG 체인을 생성합니다...")
prompt = ChatPromptTemplate.from_template("""
주어진 {context}의 내용을 바탕으로 다음 질문에 답변해 주세요:

질문: {input}
""")
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
qa_chain = create_retrieval_chain(retriever, document_chain)
print("RAG 체인 생성 완료.")

# 6. 챗봇 실행 - 사용자 질문에 답변하기
question = "우리 동아리 MT 언제 가?"
response = qa_chain.invoke({"input": question})

# (중요) 최신 방식에서는 답변이 'result'가 아닌 'answer' 키에 저장됩니다.
print(f"질문: {question}")
print(f"답변: {response['answer']}")

question_2 = "회비는 얼마고 어디로 내야해?"
response_2 = qa_chain.invoke({"input": question_2})
print(f"\n질문: {question_2}")
print(f"답변: {response_2['answer']}")