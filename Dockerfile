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

# Install dependencies as root# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variables for the OpenEnv validator
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Expose the port
EXPOSE 7860


# Entry point for the Gradio dashboard
CMD ["python", "app.py"]
