from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv("C:/Users/변상원/SAI/.env")
assert os.getenv("GOOGLE_API_KEY")

# 모델 후보: "gemini-1.5-flash-002" 또는 "gemini-1.5-flash-latest"
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002", temperature=0.7)
print(llm.invoke("OK라고만 답해.").content)
