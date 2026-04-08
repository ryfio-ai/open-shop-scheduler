# Use a compatible Python image (Gradio 5 requires 3.10+)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PATH="/home/user/.local/bin:$PATH"

# Create a non-root user for security and HF compliance
RUN useradd -m -u 1000 user

# Set work directory
WORKDIR /app

# Install dependencies as root for caching
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Switch to the non-root user
USER user

# Copy the rest of the application
COPY --chown=user:user . .

# Expose the standard HF port
EXPOSE 7860

# Ensure inference script is executable
RUN chmod +x inference.py

# Entry point for the Gradio dashboard
CMD ["python", "app.py"]
