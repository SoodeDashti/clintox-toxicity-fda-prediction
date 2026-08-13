# ------------------------------------------------------------------
# FastAPI application exposing the ClinTox multi-task model as a
# web service. Every prediction request is logged to PostgreSQL,
# acting as a simple audit trail / monitoring log. Also exposes a
# natural-language /ask endpoint backed by a local Ollama agent.
# ------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import init_db, get_db, Prediction
from app.model import predict
from app.schemas import PredictionRequest, PredictionResponse
from app.agent import ask_agent

app = FastAPI(
    title="ClinTox Toxicity & FDA Approval Predictor",
    description="Predicts clinical trial toxicity risk and FDA approval "
                "likelihood for a molecule given its SMILES string.",
    version="1.0.0",
)


@app.on_event("startup")
def on_startup():
    """Ensure the predictions table exists before serving requests."""
    init_db()


@app.get("/")
def root():
    return {"status": "ok", "message": "ClinTox prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict_toxicity(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Predict FDA approval likelihood and clinical toxicity risk for a
    given molecule, and log the request/result to the database.
    """
    result = predict(request.smiles)

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid SMILES string — RDKit could not parse it."
        )

    log_entry = Prediction(
        smiles=request.smiles,
        fda_approved_prob=result["fda_approved_prob"],
        ct_tox_prob=result["ct_tox_prob"],
        aromatic_amine_alert=result["aromatic_amine_alert"],
    )
    db.add(log_entry)
    db.commit()

    return PredictionResponse(smiles=request.smiles, **result)


@app.get("/predictions/history")
def get_history(limit: int = 20, db: Session = Depends(get_db)):
    """Return the most recent logged predictions."""
    records = (
        db.query(Prediction)
        .order_by(Prediction.predicted_at.desc())
        .limit(limit)
        .all()
    )
    return records


class AgentQuestion(BaseModel):
    question: str


@app.post("/ask")
def ask(request: AgentQuestion):
    """Ask the local agent a natural-language question about a molecule."""
    answer = ask_agent(request.question)
    return {"question": request.question, "answer": answer}