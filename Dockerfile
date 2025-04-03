FROM python:3.10.6-buster

COPY requirements.txt .
COPY setup.py .
COPY models/ ./models/
COPY raw_data/model_pipeline.pkl ./raw_data/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "models.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
