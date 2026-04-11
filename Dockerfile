FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# non-root user required by HF Spaces
RUN useradd -m -u 1000 user
WORKDIR /app

# install deps first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything and fix ownership
COPY --chown=user:user . .

USER user
EXPOSE 7860

# ── KEY: run server/app.py directly ──────────────────────────────
CMD ["python", "server/app.py"]
