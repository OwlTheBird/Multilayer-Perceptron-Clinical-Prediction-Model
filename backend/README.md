# Clinical Prediction API Backend

FastAPI backend serving predictions from the ML models.

## Setup

```bash
cd backend
pip install -r requirements.txt
```

## Run Server

```bash
python main.py
# or
uvicorn main:app --reload
```

Server runs at: http://localhost:8000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/models` | GET | List available models |
| `/features` | GET | List required features |
| `/predict/classification` | POST | Kidney disease class |
| `/predict/regression` | POST | Kidney ACR value |
| `/predict/mtl` | POST | Multi-task predictions |

## API Docs

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Request

```bash
curl -X POST http://localhost:8000/predict/classification \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, 0.3, ..., 0.0]}'
```
