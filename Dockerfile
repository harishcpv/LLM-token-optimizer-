FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the exact application/test dependency set before copying source so
# rebuilds can reuse the dependency layer.
COPY requirements.lock pyproject.toml ./
RUN python -m pip install --no-cache-dir -r requirements.lock

COPY src ./src
COPY app.py audit_cli.py README.md ./
RUN python -m pip install --no-cache-dir --no-deps .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
