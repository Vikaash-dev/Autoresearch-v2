FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml .
COPY autoresearch/ autoresearch/
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["autoresearch"]
