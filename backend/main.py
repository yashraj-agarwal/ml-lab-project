from pathlib import Path
import json
import ast

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .predict import analyze

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "models" / "metrics.json"
app = FastAPI(title="CodeTrace AI", version="1.0.0", description="Static ML-based code style estimate")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET", "POST"], allow_headers=["*"])

class AnalyzeRequest(BaseModel):
    code: str = Field(min_length=20, max_length=20000)
    language: str = "python"

@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": (ROOT / "models" / "codetrace.joblib").exists()}

@app.get("/api/metrics")
@app.get("/api/models")
def metrics():
    if not METRICS.exists():
        raise HTTPException(503, "Model metrics are unavailable. Train the project first.")
    return json.loads(METRICS.read_text(encoding="utf-8"))

@app.post("/api/analyze")
def analyze_code(request: AnalyzeRequest):
    if request.language.lower() != "python":
        raise HTTPException(422, "Only Python is currently supported.")
    if not request.code.strip():
        raise HTTPException(422, "Paste Python source code to analyze.")
    try:
        ast.parse(request.code)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise HTTPException(422, "Submit syntactically valid Python source code.") from exc
    try:
        return analyze(request.code)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
