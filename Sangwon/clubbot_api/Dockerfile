# 1. 공식 Python 이미지 사용 (버전은 3.10 예시)
FROM python:3.10-slim

# 2. 컨테이너 안 작업 디렉토리 설정
WORKDIR /app

# 3. requirements 먼저 복사 후 설치
#   (캐시를 위해 코드를 복사하기 전에 하는 게 좋음)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 4. 나머지 앱 코드 복사
COPY . .

# 5. 컨테이너가 뜰 때 실행할 명령
#   0.0.0.0 으로 바인딩해야 외부에서 접속 가능
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
