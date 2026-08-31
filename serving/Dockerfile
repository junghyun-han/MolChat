# MolChat serving image — FastAPI + the offline rule-based agent (no API key).
FROM python:3.12-slim

WORKDIR /app

# System libs RDKit wheels need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt fastapi "uvicorn[standard]"

COPY molchat ./molchat
COPY serving ./serving
COPY data ./data

ENV MOLCHAT_LLM=rule-based
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
