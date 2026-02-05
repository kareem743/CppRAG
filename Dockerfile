FROM ollama/ollama:latest AS ollama

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ollama /usr/bin/ollama /usr/local/bin/ollama

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml setup.py rag_core.cpp rag_cli.py /app/
COPY rag /app/rag
RUN pip install --no-cache-dir .

COPY . /app

ENTRYPOINT ["python", "rag_cli.py"]
CMD ["--help"]
