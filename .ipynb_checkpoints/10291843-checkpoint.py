import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 0. Google API 키 설정 (사전에 발급받은 키를 환경 변수로 설정)
# os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY"

# 1. 지식 베이스(텍스트 파일) 로드
with open('club_info.txt', 'r', encoding='utf-8') as f:
    club_info = f.read()

# 2. 텍스트를 작은 조각(Chunk)으로 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_text(club_info)

# 3. Google 임베딩 모델 로드 및 벡터 DB 생성
# 텍스트 조각들을 벡터로 변환하고, FAISS 데이터베이스에 저장합니다.
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_texts(texts, embeddings)

# 4. Google Gemini Pro 모델 로드
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.7, # 창의성 조절 (0에 가까울수록 사실 기반)
                             convert_system_message_to_human=True) # 시스템 메시지를 사람의 메시지처럼 변환

# 5. RAG 체인(Chain) 생성
# retriever: 질문과 관련된 문서를 벡터 DB에서 찾아주는 역할
# llm: retriever가 찾은 정보를 바탕으로 답변을 생성하는 역할
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 찾은 문서들을 모두 프롬프트에 넣는 방식
    retriever=vectorstore.as_retriever()
)

# 6. 챗봇 실행 - 사용자 질문에 답변하기
question = "우리 동아리 MT 언제 가?"
response = qa_chain.invoke({"query": question})

print(f"질문: {response['query']}")
print(f"답변: {response['result']}")

question_2 = "회비는 얼마고 어디로 내야해?"
response_2 = qa_chain.invoke({"query": question_2})
print(f"\n질문: {response_2['query']}")
print(f"답변: {response_2['result']}")