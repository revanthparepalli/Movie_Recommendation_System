FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

LABEL maintainer="revanth.parepalli@convergytics.com"

COPY . /app

WORKDIR /app

RUN pip install --upgrade pip && pip install -r requirements.txt