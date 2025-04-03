FROM python:3.10.6-buster

COPY requirements.txt .
COPY setup.py .
COPY models/ ./models/
COPY raw_data/model_pipeline.pkl ./raw_data/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["uvicorn", "models.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]





# FROM python:3.10.6-buster

# COPY requirements.txt /requirements.txt
# COPY setup.py setup.py

# RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

# COPY models/ml_logic/model.py ./models/ml_logic/
# COPY raw_data/model_pipeline.pkl ./raw_data/

# CMD uvicorn models.api.fast:app --host 0.0.0.0 --port $PORT
# CMD ["uvicorn", "models.api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
