# ────────────────────────────────────────────────────────
# Slim image for serving predictions with FastAPI
# ────────────────────────────────────────────────────────
FROM python:3.10-slim

# 1) Set workdir
WORKDIR /app

# 2) Install Python dependencies
COPY requirement.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirement.txt

# 3) Copy inference code
COPY src/ ./src/

# 4) Copy the trained pipeline & feature list
#    (assumes you ran the training container and populated Models/)
RUN mkdir -p models
COPY Models/best_pipeline.pkl models/best_pipeline.pkl
COPY selected_features.txt .

# 5) Expose the API port
EXPOSE 8000

# 6) Launch FastAPI
CMD ["uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8000"]