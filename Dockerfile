FROM python:3.10.6-buster

COPY requirements.txt .
COPY models/ml_logic/model.py ./models/ml_logic/
COPY raw_data/model_pipeline.pkl ./raw_data/

RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# CMD uvicorn models.api.fast:app --host 0.0.0.0 --port $PORT
CMD ["uvicorn", "fast:app", "--host", "0.0.0.0", "--port", "8000"]
