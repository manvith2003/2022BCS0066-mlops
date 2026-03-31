FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# In a real pipeline, the training step creates model.pkl and might even pull data by DVC,
# but for the API, we only need the API code and the trained model artifact.
COPY src/api.py src/
COPY model.pkl . 

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
