FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

RUN pip install --no-cache-dir -r requirements.txt --timeout 1000 --retries 10

EXPOSE 8000

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]