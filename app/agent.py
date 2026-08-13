# ------------------------------------------------------------------
# Simple tool-calling agent using a local Ollama model. The agent
# can call two tools: predicting toxicity/FDA approval for a SMILES
# string (using our own model), and looking up basic compound info
# from PubChem by name (e.g. "aspirin" -> SMILES).
#
# Runs inside Docker, so it connects to Ollama on the host machine
# via "host.docker.internal" rather than "localhost".
# ------------------------------------------------------------------

import httpx
import json

from ollama import Client
from app.model import predict as predict_toxicity

OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_HOST = "http://host.docker.internal:11434"

client = Client(host=OLLAMA_HOST)

SYSTEM_PROMPT = (
    "You are a toxicology assistant. For any question about a specific "
    "molecule's toxicity or FDA approval likelihood, you MUST use the "
    "available tools (looking up the SMILES via PubChem if needed, then "
    "calling the prediction tool) rather than relying on your own general "
    "medical knowledge. Do not invent dosage calculators, code, or medical "
    "advice that isn't grounded in the tool results."
)


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

        smiles_url = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            f"{cid}/property/CanonicalSMILES/JSON"
        )
        smiles_response = httpx.get(smiles_url, timeout=10)
        smiles_response.raise_for_status()
        smiles = smiles_response.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]

        return {"compound_name": compound_name, "smiles": smiles, "found": True}
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


def ask_agent(question: str) -> str:
    """
    Send a natural-language question to the local Ollama model, let
    it call tools as needed (PubChem lookup, toxicity prediction),
    and return a final natural-language answer.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    response = client.chat(model=OLLAMA_MODEL, messages=messages, tools=TOOLS)

    message = response.message if hasattr(response, "message") else response["message"]
    tool_calls = getattr(message, "tool_calls", None) or message.get("tool_calls")

    messages.append({
        "role": "assistant",
        "content": getattr(message, "content", None) or message.get("content", ""),
    })

    if tool_calls:
        for tool_call in tool_calls:
            func = tool_call.function if hasattr(tool_call, "function") else tool_call["function"]
            func_name = getattr(func, "name", None) or func["name"]
            func_args = getattr(func, "arguments", None) or func["arguments"]

            if func_name in AVAILABLE_FUNCTIONS:
                result = AVAILABLE_FUNCTIONS[func_name](**func_args)
            else:
                result = {"error": f"Unknown tool: {func_name}"}

            messages.append({
                "role": "tool",
                "content": json.dumps(result),
            })

        final_response = client.chat(model=OLLAMA_MODEL, messages=messages)
        final_message = final_response.message if hasattr(final_response, "message") else final_response["message"]
        return getattr(final_message, "content", None) or final_message.get("content", "")

    return getattr(message, "content", None) or message.get("content", "")