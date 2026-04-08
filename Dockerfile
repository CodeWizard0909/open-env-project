FROM python:3.11-slim

# Required for Hugging Face Spaces
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first (for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package source
COPY hospital_bed_env/ ./hospital_bed_env/
COPY pyproject.toml .
COPY README.md .
COPY openenv.yaml .

# Install the package itself in editable mode
RUN pip install --no-cache-dir -e .

# Expose the standard Hugging Face Spaces port
EXPOSE 7860

# Run the FastAPI server using uvicorn
CMD ["uvicorn", "hospital_bed_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
