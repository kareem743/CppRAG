FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py /app/
RUN pip install --no-cache-dir .

COPY . /app
RUN pip install --no-cache-dir .

CMD ["tail", "-f", "/dev/null"]
