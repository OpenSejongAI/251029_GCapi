### instruction

1. environment.yml을 통해 콘다환경 구축
    - conda env create -f environment.yml (터미널에 명령어 입력하여 'opsi' 환경 생성)
    - conda activate opsi (위 명령어를 입력하여 가상환경 활성화, 에디터에서 기본 가상환경 지정 가능)
2. gcloud cli 환경 검사
    - gcloud config list (내 계정 확인)
    - gcloud services list --enabled (활성화된 API 목록 확인)
    - gcloud ai endpoints list --region=us-central1 (vertex ai 엔드포인트 설정확인)
3. rag_data 폴더 생성 후 그 안에 rag가 참조할 문서 첨부
4. data_loader python파일을 통한 rag_data 내 문서 탐색/읽기
5. Docker 배포
