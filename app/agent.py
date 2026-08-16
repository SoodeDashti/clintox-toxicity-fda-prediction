# ------------------------------------------------------------------
# Simple tool-calling agent using a local Ollama model. The agent
# can call two tools: predicting toxicity/FDA approval for a SMILES
# string (using our own model), and looking up basic compound info
# from PubChem by name (e.g. "aspirin" -> SMILES).
#
# Runs inside Docker, so it connects to Ollama on the host machine
# via "host.docker.internal" rather than "localhost".
#
# The final answer is built deterministically in Python from the raw
# tool outputs, rather than trusting the local 8B model to restate
# SMILES strings/numbers accurately — it has been observed to
# paraphrase tool results incorrectly (e.g. inventing a wrong SMILES).
# The LLM is only used to decide which tool(s) to call.
# ------------------------------------------------------------------

import httpx
import json
from ollama import Client

from app.model import predict as predict_toxicity

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://host.docker.internal:11434"

ollama_client = Client(host=OLLAMA_HOST)


def lookup_pubchem_by_name(compound_name: str) -> dict:
    """
    Look up a compound's canonical SMILES from PubChem, given a
    common name (e.g. "aspirin", "caffeine").
    """
    try:
        cid_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
            f"{compound_name}/cids/JSON"
        )
        cid_response = httpx.get(cid_url, timeout=10)
        cid_response.raise_for_status()
        cid = cid_response.json()["IdentifierList"]["CID"][0]

        # PubChem has renamed CanonicalSMILES to ConnectivitySMILES in
        # some API responses; request both and accept whichever is present.
        smiles_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/property/ConnectivitySMILES,CanonicalSMILES/JSON"
        )
        smiles_response = httpx.get(smiles_url, timeout=10)
        smiles_response.raise_for_status()
        properties = smiles_response.json()["PropertyTable"]["Properties"][0]
        smiles = properties.get("CanonicalSMILES") or properties.get("ConnectivitySMILES")

        return {"compound_name": compound_name, "smiles": smiles, "found": smiles is not None}
    except Exception:
        return {"compound_name": compound_name, "smiles": None, "found": False}


def predict_toxicity_tool(smiles: str) -> dict:
    """Run our trained model's prediction for a given SMILES string."""
    result = predict_toxicity(smiles)
    if result is None:
        return {"error": "Invalid SMILES string, could not parse."}
    return result


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_pubchem_by_name",
            "description": "Look up a compound's SMILES string from PubChem given its common name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "compound_name": {"type": "string", "description": "Common name of the compound, e.g. 'aspirin'"}
                },
                "required": ["compound_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_toxicity_tool",
            "description": "Predict FDA approval probability and clinical toxicity probability for a molecule given its SMILES string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "SMILES string of the molecule"}
                },
                "required": ["smiles"],
            },
        },
    },
]

AVAILABLE_FUNCTIONS = {
    "lookup_pubchem_by_name": lookup_pubchem_by_name,
    "predict_toxicity_tool": predict_toxicity_tool,
}

SYSTEM_PROMPT = """You are a molecular toxicity assistant. You have access to two tools:
1. lookup_pubchem_by_name - looks up a compound's SMILES string from PubChem
2. predict_toxicity_tool - predicts FDA approval and clinical toxicity probability for a SMILES string

When the user asks about a compound by name, call lookup_pubchem_by_name first
to get its SMILES, then call predict_toxicity_tool with that exact SMILES.
"""


def ask_agent(question: str) -> str:
    """
    Send a natural-language question to the local Ollama model, let
    it decide which tool(s) to call, execute those tools, and return
    a deterministic answer built directly from the real tool outputs
    (not from the model's own restatement of them).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    response = ollama_client.chat(model=OLLAMA_MODEL, messages=messages, tools=TOOLS)
    message = response.message if hasattr(response, "message") else response["message"]
    tool_calls = getattr(message, "tool_calls", None) or message.get("tool_calls")

    if not tool_calls:
        return getattr(message, "content", None) or message.get("content", "")

    # Execute tools and keep the real results in Python, not trusting
    # the model to repeat them verbatim later.
    tool_results = []
    for tool_call in tool_calls:
        func = tool_call.function if hasattr(tool_call, "function") else tool_call["function"]
        func_name = getattr(func, "name", None) or func["name"]
        func_args = getattr(func, "arguments", None) or func["arguments"]

        if func_name in AVAILABLE_FUNCTIONS:
            result = AVAILABLE_FUNCTIONS[func_name](**func_args)
        else:
            result = {"error": f"Unknown tool: {func_name}"}

        tool_results.append((func_name, func_args, result))

    # If a name lookup succeeded but toxicity wasn't predicted yet
    # (small models sometimes only call one tool per turn), chain the
    # second call ourselves using the real SMILES we just retrieved.
    has_lookup = any(name == "lookup_pubchem_by_name" for name, _, _ in tool_results)
    has_prediction = any(name == "predict_toxicity_tool" for name, _, _ in tool_results)

    if has_lookup and not has_prediction:
        for name, _, result in tool_results:
            if name == "lookup_pubchem_by_name" and result.get("found"):
                pred_result = predict_toxicity_tool(result["smiles"])
                tool_results.append(("predict_toxicity_tool", {"smiles": result["smiles"]}, pred_result))
                break

    # Build a deterministic, guaranteed-accurate summary directly from
    # the real tool outputs — no LLM paraphrasing of numbers/SMILES.
    lines = []
    for func_name, func_args, result in tool_results:
        if func_name == "lookup_pubchem_by_name":
            if result.get("found"):
                lines.append(f"**{result['compound_name']}** → SMILES: `{result['smiles']}`")
            else:
                lines.append(f"Could not find '{func_args.get('compound_name')}' on PubChem.")
        elif func_name == "predict_toxicity_tool":
            if "error" in result:
                lines.append(f"Prediction error: {result['error']}")
            else:
                lines.append(
                    f"- FDA approval probability: **{result['fda_approved_prob']:.1%}**\n"
                    f"- Clinical toxicity probability: **{result['ct_tox_prob']:.1%}**\n"
                    f"- Aromatic amine structural alert: **{'Yes' if result['aromatic_amine_alert'] else 'No'}**"
                )

    return "\n\n".join(lines)