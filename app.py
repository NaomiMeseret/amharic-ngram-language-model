from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.ngram_model import NGramModel

MODEL_DIR = Path(__file__).resolve().parent / "results" / "model"


class PredictRequest(BaseModel):
    text: str = ""
    top_k: int = Field(default=5, ge=1, le=10)


class PredictResponse(BaseModel):
    predictions: list[dict[str, Any]]


class GenerateRequest(BaseModel):
    prompt: str = ""
    max_tokens: int = Field(default=5, ge=1, le=20)


class GenerateResponse(BaseModel):
    generated_text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = NGramModel.load(str(MODEL_DIR))
    yield
    app.state.model = None


app = FastAPI(title="Amharic N-gram Model API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|0\.0\.0\.0):5173",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sorted_predictions(text: str, top_k: int) -> list[dict[str, Any]]:
    model: NGramModel = app.state.model
    if not text or not text.strip():
        return []

    predictions = model.predict_next(text, k=top_k, method="backoff")
    predictions = sorted(predictions, key=lambda item: item[1], reverse=True)

    return [{"word": word, "prob": float(prob)} for word, prob in predictions[:top_k]]


def _generate_text(model: NGramModel, prompt: str, max_tokens: int) -> str:
    text = (prompt or "").strip()
    if not text:
        suggestions = model.predict_next("", k=1, method="backoff")
        if not suggestions:
            return ""
        tokens = [suggestions[0][0]]
    else:
        tokens = text.split()

    for _ in range(max_tokens):
        context = " ".join(tokens[-4:]) if tokens else ""
        suggestions = model.predict_next(context, k=1, method="backoff")
        if not suggestions:
            break
        next_word = suggestions[0][0]
        tokens.append(next_word)

    return " ".join(tokens).strip()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    return {"predictions": _sorted_predictions(request.text, request.top_k)}


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    model: NGramModel = app.state.model
    generated = _generate_text(model, request.prompt, request.max_tokens)
    return {"generated_text": generated}
