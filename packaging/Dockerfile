# Vertex AI Kakao Skill Flask server (packaging/)
FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy conda env + project files
COPY environment.yml ./environment.yml
COPY packaging/ ./packaging/
COPY ../rag_data ./rag_data/

# Create conda env defined in packaging/environment.yml
RUN apt-get update && apt-get install -y build-essential
RUN conda update -n base conda -y
RUN conda install -n base pip setuptools -y
RUN conda env create -f environment.yml

# Use the new env (name: opsi per yml)
SHELL ["conda", "run", "-n", "opsi", "/bin/bash", "-c"]

# Default Flask vars + port
ENV FLASK_APP=packaging.kakao_skill_api:app \
    PORT=5000

EXPOSE 5000

# Run the Flask Kakao Skill API
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
