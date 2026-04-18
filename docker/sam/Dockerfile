FROM python:3.10-slim
WORKDIR /app

# Install system deps required for building Python packages and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
	git \
	build-essential \
	libgl1 \
	libglib2.0-0 \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements-sam.txt ./
RUN pip install --no-cache-dir -r requirements-sam.txt

COPY app_sam.py ./
EXPOSE 8000
CMD ["uvicorn","app_sam:app","--host","0.0.0.0","--port","8000"]
