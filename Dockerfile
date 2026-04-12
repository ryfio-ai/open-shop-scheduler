FROM ghcr.io/meta-pytorch/openenv-base:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install as editable package
RUN pip install -e .

# non-root user required by HF Spaces
RUN useradd -m -u 1000 user 2>/dev/null || true
USER user

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
