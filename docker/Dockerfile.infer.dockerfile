# Dockerfile.infer
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy inference service code and the saved pipeline
COPY src/ ./src/
COPY models/best_pipeline.pkl ./models/best_pipeline.pkl
COPY selected_features.txt .

# Expose port and run FastAPI
EXPOSE 8000
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]