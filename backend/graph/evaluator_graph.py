from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
import os, json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class EvaluatorState(TypedDict):
    doctor_message: str
    patient_history: List[str]
    evaluation: Dict


def evaluator_node(state: EvaluatorState):
    prompt = f"""
Patient history:
{state["patient_history"]}

Doctor's question:
"{state["doctor_message"]}"

Classify the question.
Respond ONLY in JSON.
"""

    response = client.chat.completions.create(
        model="groq/compound",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        evaluation = json.loads(response.choices[0].message.content)
    except:
        evaluation = {"verdict": "WARN", "reason": "Invalid JSON"}

    return {
        "doctor_message": state["doctor_message"],
        "patient_history": state["patient_history"],
        "evaluation": evaluation
    }


builder = StateGraph(EvaluatorState)
builder.add_node("evaluator", evaluator_node)
builder.set_entry_point("evaluator")
builder.add_edge("evaluator", END)

evaluator_graph = builder.compile()

