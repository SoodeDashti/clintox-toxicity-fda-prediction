# ------------------------------------------------------------------
# Pydantic models defining the shape of API requests and responses.
# ------------------------------------------------------------------

from pydantic import BaseModel


class PredictionRequest(BaseModel):
    smiles: str

    class Config:
        json_schema_extra = {
            "example": {"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}  # aspirin
        }


class PredictionResponse(BaseModel):
    smiles: str
    fda_approved_prob: float
    ct_tox_prob: float
    aromatic_amine_alert: bool
