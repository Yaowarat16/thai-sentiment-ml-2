import os
import time
import joblib

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# =========================
# APP CONFIG
# =========================
app = FastAPI(
    title="Thai Sentiment Analysis",
    description="Thai Sentiment Classification with A/B Model Comparison",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(BASE_DIR, "app", "templates")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

templates = Jinja2Templates(directory=TEMPLATE_DIR)

# =========================
# LOAD MODELS
# =========================
MODEL_A_PATH = os.path.join(OUTPUT_DIR, "LogisticRegression.joblib")
MODEL_B_PATH = os.path.join(OUTPUT_DIR, "LinearSVM.joblib")

bundle_a = joblib.load(MODEL_A_PATH)
bundle_b = joblib.load(MODEL_B_PATH)

model_a = bundle_a["pipeline"]
model_b = bundle_b["pipeline"]

MODEL_A_NAME = bundle_a.get("model_name", "Logistic Regression")
MODEL_B_NAME = bundle_b.get("model_name", "Linear SVM")

# =========================
# SCHEMA
# =========================
class TextInput(BaseModel):
    text: str

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_a": MODEL_A_NAME,
            "model_b": MODEL_B_NAME
        }
    )

@app.post("/predict")
def predict(data: TextInput):
    text = data.text

    start = time.perf_counter()
    label = model_a.predict([text])[0]
    latency = (time.perf_counter() - start) * 1000

    confidence = None
    if hasattr(model_a, "predict_proba"):
        confidence = float(model_a.predict_proba([text])[0].max())

    return JSONResponse({
        "label": label,
        "confidence": round(confidence, 4) if confidence is not None else None,
        "latency_ms": round(latency, 2),
        "name": MODEL_A_NAME
    })

@app.post("/predict_ab")
def predict_ab(data: TextInput):
    text = data.text

    # ---- Model A
    t0 = time.perf_counter()
    pa = model_a.predict([text])[0]
    la = (time.perf_counter() - t0) * 1000

    ca = None
    if hasattr(model_a, "predict_proba"):
        ca = float(model_a.predict_proba([text])[0].max())

    # ---- Model B
    t0 = time.perf_counter()
    pb = model_b.predict([text])[0]
    lb = (time.perf_counter() - t0) * 1000

    # LinearSVM has no predict_proba → use decision score
    cb = None
    if hasattr(model_b, "decision_function"):
        score = model_b.decision_function([text])
        cb = float(abs(score).max())
        cb = min(cb / (cb + 1), 1.0)  # normalize to 0–1

    return JSONResponse({
        "model_a": {
            "label": pa,
            "confidence": round(ca, 4) if ca is not None else None,
            "latency_ms": round(la, 2),
            "name": MODEL_A_NAME
        },
        "model_b": {
            "label": pb,
            "confidence": round(cb, 4) if cb is not None else None,
            "latency_ms": round(lb, 2),
            "name": MODEL_B_NAME
        }
    })

@app.get("/model/info")
def model_info():
    return {
        "model_a": MODEL_A_NAME,
        "model_b": MODEL_B_NAME
    }

@app.get("/health")
def health():
    return {"status": "ok"}
