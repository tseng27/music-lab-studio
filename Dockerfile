# 音乐实验室工作站 — one container serves the studio page and the AI API.
# Works as a Hugging Face Docker Space (port 7860). Pick a GPU for real models.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# HF Spaces run as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# INSTALL_AI=0 builds a light image (demo mode only); default installs the models' libraries
ARG INSTALL_AI=1
COPY backend/requirements.txt backend/requirements-ai.txt backend/
RUN pip install --no-cache-dir -r backend/requirements.txt && \
    if [ "$INSTALL_AI" = "1" ]; then pip install --no-cache-dir -r backend/requirements-ai.txt; fi

COPY --chown=user . .
USER user
ENV HOME=/home/user MUSICLAB_DATA=/home/user/data HF_HOME=/home/user/.cache/huggingface
EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
