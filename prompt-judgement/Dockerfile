FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime as base


RUN apt-get update && apt-get install build-essential -y --no-install-recommends  && rm -rf /var/lib/apt/lists/*

WORKDIR /build

ADD requirements.txt .

RUN pip install --upgrade pip --no-cache-dir \ 
    && pip install -r requirements.txt --no-cache-dir \ 
    && rm -f requirements.txt

WORKDIR /app

ADD app.py .

ENTRYPOINT [ "python3", "app.py" ]