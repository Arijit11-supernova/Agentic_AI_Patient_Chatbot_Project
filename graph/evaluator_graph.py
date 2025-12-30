# evaluator_graph.py

from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
import os, json, logging

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

logging.basicConfig(level=logging.INFO)

class EvaluatorState(TypedDict):
    doctor_message: str
    patient_history: List[str]
    evaluation: Dict

def evaluator_node(state: EvaluatorState):
    """
    Evaluates the doctor's question against patient history using Groq.
    Returns updated state with evaluation results.
    """
    # Convert patient_history list to plain text
    history_text = "\n".join(state.get("patient_history", []))
    prompt = f"""
Patient history:
{history_text}

Doctor's question:
"{state.get("doctor_message", "")}"

Classify the question.
Respond ONLY in JSON.
"""

    try:
        response = client.chat.completions.create(
            model="groq/compound",
            messages=[{"role": "user", "content": prompt}]
        )

        raw_content = response.choices[0].message.content
        try:
            evaluation = json.loads(raw_content)
        except json.JSONDecodeError:
            logging.warning(f"Invalid JSON from Groq: {raw_content}")
            evaluation = {"verdict": "WARN", "reason": "Invalid JSON from Groq"}

    except Exception as e:
        logging.error(f"Groq API error: {e}")
        evaluation = {"verdict": "ERROR", "reason": str(e)}

    return {
        "doctor_message": state.get("doctor_message", ""),
        "patient_history": state.get("patient_history", []),
        "evaluation": evaluation
    }

# Build the state graph
builder = StateGraph(EvaluatorState)
builder.add_node("evaluator", evaluator_node)
builder.set_entry_point("evaluator")
builder.add_edge("evaluator", END)
evaluator_graph = builder.compile()


