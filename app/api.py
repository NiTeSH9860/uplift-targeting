import os
from contextlib import asynccontextmanager
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "models/s_learner_criteo_visit.joblib")
MAX_BATCH = 1000
FEATURES = [f"f{i}" for i in range(12)]

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once at startup, not on every request
    artifact = joblib.load(MODEL_PATH)
    if artifact["features"] != FEATURES:
        raise RuntimeError("Model features do not match the API schema")
    state["artifact"] = artifact
    yield
    state.clear()


app = FastAPI(
    title="Uplift Targeting API",
    description="Scores customers by predicted campaign uplift and recommends who to contact.",
    version="0.1.0",
    lifespan=lifespan,
)


class Customer(BaseModel):
    f0: float
    f1: float
    f2: float
    f3: float
    f4: float
    f5: float
    f6: float
    f7: float
    f8: float
    f9: float
    f10: float
    f11: float


class ScoreRequest(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=MAX_BATCH)


class ScoreItem(BaseModel):
    uplift: float = Field(description="Predicted change in outcome probability if contacted")
    top_percent: float = Field(description="Customer ranks in the top X% by uplift (lower is better)")


class ScoreResponse(BaseModel):
    results: List[ScoreItem]


class RecommendRequest(BaseModel):
    customers: List[Customer] = Field(min_length=1, max_length=MAX_BATCH)
    value: float = Field(gt=0, description="Profit from one incremental outcome")
    cost: float = Field(ge=0, description="Cost of contacting one customer")
    max_contacts: Optional[int] = Field(default=None, ge=0,
                                        description="Optional cap on how many customers to contact")


class RecommendItem(BaseModel):
    uplift: float
    top_percent: float
    expected_profit: float = Field(description="value * uplift - cost, if this customer is contacted")
    contact: bool


class RecommendResponse(BaseModel):
    n_customers: int
    n_contact: int
    expected_total_profit: float
    results: List[RecommendItem]


def _score(customers):
    """Return (uplift, top_percent) arrays for a list of Customer objects."""
    artifact = state["artifact"]
    X = np.array([[getattr(c, f) for f in artifact["features"]] for c in customers], dtype=float)
    if not np.isfinite(X).all():
        raise HTTPException(status_code=422, detail="Features must be finite numbers")

    uplift = artifact["model"].predict_uplift(X)
    q = artifact["score_quantiles"]
    # Where does this score fall among held-out test customers?
    pct = np.interp(uplift, q, np.linspace(0, 100, len(q)))
    return uplift, 100 - pct


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    a = state["artifact"]
    q = a["score_quantiles"]
    return {
        "target": a["target"],
        "n_train": a["n_train"],
        "trained_at": a["trained_at"],
        "features": a["features"],
        "score_median": round(float(q[50]), 5),
        "score_p90": round(float(q[90]), 5),
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest):
    uplift, top = _score(req.customers)
    return ScoreResponse(results=[
        ScoreItem(uplift=round(float(u), 5), top_percent=round(float(t), 1))
        for u, t in zip(uplift, top)
    ])


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    uplift, top = _score(req.customers)
    profit = req.value * uplift - req.cost

    # Only customers with positive expected profit are candidates
    chosen = np.where(profit > 0)[0]
    if req.max_contacts is not None:
        chosen = chosen[np.argsort(-profit[chosen])][: req.max_contacts]

    contact = np.zeros(len(uplift), dtype=bool)
    contact[chosen] = True

    return RecommendResponse(
        n_customers=len(uplift),
        n_contact=int(contact.sum()),
        expected_total_profit=round(float(profit[contact].sum()), 5),
        results=[
            RecommendItem(uplift=round(float(u), 5), top_percent=round(float(t), 1),
                          expected_profit=round(float(p), 5), contact=bool(c))
            for u, t, p, c in zip(uplift, top, profit, contact)
        ],
    )

from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")
