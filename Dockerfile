# Use a slim Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the Gradio port
EXPOSE 7860

# Ensure the app is runnable
RUN chmod +x inference.py

# Entry point for HF Spaces
ENTRYPOINT ["python", "app.py"]
