FROM python:3.11-slim

# System dependencies needed for RDKit to run properly
RUN apt-get update && apt-get install -y \
    libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY models/ ./models/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]