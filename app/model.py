# ------------------------------------------------------------------
# Loads the trained multi-task model and turns a raw SMILES string
# into a prediction, using the same featurization pipeline (Morgan
# fingerprints + aromatic amine structural alert) built in the
# notebook. Keeping this logic identical to training time is
# essential — a mismatch here would silently break predictions.
# ------------------------------------------------------------------

import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

MODEL_PATH = "models/ct_tox_fda_model.pkl"

# Same aromatic amine SMARTS pattern used during EDA / structural alerts
AROMATIC_AMINE_SMARTS = "c[NX3;H2,H1;!$(NC=O)]"
_aromatic_amine_pattern = Chem.MolFromSmarts(AROMATIC_AMINE_SMARTS)

_model = None


def load_model():
    """Load the trained model once and cache it in memory."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def featurize_smiles(smiles: str, radius: int = 2, n_bits: int = 2048):
    """
    Convert a SMILES string into the same Morgan fingerprint
    representation used during training. Returns None if the SMILES
    string cannot be parsed by RDKit.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    fingerprint = np.array(fp).reshape(1, -1)

    has_aromatic_amine = mol.HasSubstructMatch(_aromatic_amine_pattern)

    return fingerprint, has_aromatic_amine


def predict(smiles: str):
    """
    Run the full prediction pipeline for a single SMILES string.
    Returns a dict with probabilities for both tasks, or None if
    the SMILES string is invalid.
    """
    fingerprint, has_aromatic_amine = featurize_smiles(smiles)
    if fingerprint is None:
        return None

    model = load_model()
    proba = model.predict_proba(fingerprint)
    # proba is a list of arrays, one per task: [FDA_APPROVED, CT_TOX]
    fda_approved_prob = float(proba[0][0][1])
    ct_tox_prob = float(proba[1][0][1])

    return {
        "fda_approved_prob": fda_approved_prob,
        "ct_tox_prob": ct_tox_prob,
        "aromatic_amine_alert": bool(has_aromatic_amine),
    }
