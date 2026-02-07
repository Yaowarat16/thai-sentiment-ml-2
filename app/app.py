import os
import time
import joblib
import pandas as pd

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

MODEL_A_VERSION = bundle_a.get("model_version", "v1.0")
MODEL_B_VERSION = bundle_b.get("model_version", "v1.0")

# =========================
# SCHEMA
# =========================
class TextInput(BaseModel):
    text: str

# =========================
# ROUTES
# =========================

# -------------------------
# Home (Web UI)
# -------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_a": f"{MODEL_A_NAME} ({MODEL_A_VERSION})",
            "model_b": f"{MODEL_B_NAME} ({MODEL_B_VERSION})"
        }
    )

# -------------------------
# Single-model prediction (Model A)
# -------------------------
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
        "name": MODEL_A_NAME,
        "version": MODEL_A_VERSION
    })

# -------------------------
# A/B comparison (Model A vs Model B)
# -------------------------
@app.post("/predict_ab")
def predict_ab(data: TextInput):
    text = data.text

    # ---- Model A (Logistic Regression)
    t0 = time.perf_counter()
    pa = model_a.predict([text])[0]
    la = (time.perf_counter() - t0) * 1000

    ca = None
    if hasattr(model_a, "predict_proba"):
        ca = float(model_a.predict_proba([text])[0].max())

    # ---- Model B (Linear SVM)
    t0 = time.perf_counter()
    pb = model_b.predict([text])[0]
    lb = (time.perf_counter() - t0) * 1000

    # LinearSVM has no predict_proba → normalize decision score
    cb = None
    if hasattr(model_b, "decision_function"):
        score = model_b.decision_function([text])
        cb = float(abs(score).max())
        cb = min(cb / (cb + 1), 1.0)

    return JSONResponse({
        "model_a": {
            "label": pa,
            "confidence": round(ca, 4) if ca is not None else None,
            "latency_ms": round(la, 2),
            "name": MODEL_A_NAME,
            "version": MODEL_A_VERSION
        },
        "model_b": {
            "label": pb,
            "confidence": round(cb, 4) if cb is not None else None,
            "latency_ms": round(lb, 2),
            "name": MODEL_B_NAME,
            "version": MODEL_B_VERSION
        }
    })

# -------------------------
# Error Analysis Page
# -------------------------
@app.get("/errors", response_class=HTMLResponse)
def view_errors(request: Request):
    error_path = os.path.join(OUTPUT_DIR, "misclassified_10.csv")

    if not os.path.exists(error_path):
        errors = []
    else:
        df = pd.read_csv(error_path)
        errors = df.to_dict(orient="records")

    return templates.TemplateResponse(
        "errors.html",
        {
            "request": request,
            "errors": errors
        }
    )

# -------------------------
# Model metadata (for debug / report)
# -------------------------
@app.get("/model/info")
def model_info():
    return {
        "model_a": {
            "name": MODEL_A_NAME,
            "version": MODEL_A_VERSION
        },
        "model_b": {
            "name": MODEL_B_NAME,
            "version": MODEL_B_VERSION
        }
    }

# -------------------------
# Health check
# -------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
