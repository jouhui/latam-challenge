# syntax=docker/dockerfile:1.2
FROM python:3.11

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY challenge challenge
COPY models models

EXPOSE 8080
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080", "--reload", "--reload-dir", "challenge"]