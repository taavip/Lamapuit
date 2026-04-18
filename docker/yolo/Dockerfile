FROM python:3.10-slim
WORKDIR /app

COPY requirements-yolo.txt ./
RUN pip install --no-cache-dir -r requirements-yolo.txt

COPY app_yolo.py ./
EXPOSE 8001
CMD ["uvicorn","app_yolo:app","--host","0.0.0.0","--port","8001"]
