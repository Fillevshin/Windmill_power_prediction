FROM python:3.10-slim-buster
WORKDIR /windmill_power_prediction
COPY . /windmill_power_prediction
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
