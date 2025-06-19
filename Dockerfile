FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /sush

RUN pip install uv --no-cache
COPY pyproject.toml uv.lock  .
RUN uv sync

COPY . .


CMD ["uv", "run", "python", "-m", "main"]