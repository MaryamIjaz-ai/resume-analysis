# Use lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies (with higher timeout)
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy project files
COPY . .

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]