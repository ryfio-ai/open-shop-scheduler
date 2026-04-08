# Use a compatible Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT=7860

# Create a non-root user for HF compliance
RUN useradd -m -u 1000 user
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application and set ownership
COPY --chown=user:user . .

# Switch to non-root user
USER user

# Expose port
EXPOSE 7860

# Run using the new server structure
CMD ["python", "server/app.py"]
