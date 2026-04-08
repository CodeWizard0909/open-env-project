FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY . .

# Install the package itself
RUN pip install -e .

# Standard port for Hugging Face Spaces / OpenEnv server
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "hospital_bed_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
