from fastapi import FastAPI
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
import logging

from utils.session_manager import create_session, get_session, reset_session
from graph.patient_graph import patient_graph
from graph.evaluator_graph import evaluator_graph
from graph.treatment_graph import treatment_graph

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= ROOT =================
@app.get("/")
def root():
    return {"message": "Agentic AI Patient Chatbot Backend Running"}

# ================= CHAT =================
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

@app.post("/chat")
def chat(request: ChatRequest):
    # Create or fetch session
    if not request.session_id:
        session_id = create_session()
        state = get_session(session_id)
        logging.info(f"New session created: {session_id}")
    else:
        session_id = request.session_id
        state = get_session(session_id)

        if not state:
            session_id = create_session()
            state = get_session(session_id)

    if state["conversation_end"]:
        return {
            "session_id": session_id,
            "reply": "[Conversation already completed politely.]",
            "conversation_end": True
        }

    # Append doctor message
    state["messages"].append(HumanMessage(content=request.query))

    # Invoke patient graph
    new_state = patient_graph.invoke({
        "messages": state["messages"],
        "revealed_symptoms": state["revealed_symptoms"],
        "conversation_end": False
    })

    state["messages"] = new_state["messages"]
    state["revealed_symptoms"] = new_state["revealed_symptoms"]
    state["conversation_end"] = new_state["conversation_end"]

    return {
        "session_id": session_id,
        "reply": state["messages"][-1].content,
        "conversation_end": state["conversation_end"]
    }

# ================= EVALUATION =================
class EvaluationRequest(BaseModel):
    doctor_message: str
    patient_history: List[str]

@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    result = evaluator_graph.invoke({
        "doctor_message": request.doctor_message,
        "patient_history": request.patient_history
    })
    return {"evaluation": result}

# ================= TREATMENT =================
class TreatmentRequest(BaseModel):
    prescription: str
    session_id: Optional[str] = None

@app.post("/treatment")
def treatment(request: TreatmentRequest):
    if not request.session_id:
        return {
            "patient_reply": "No active session found.",
            "conversation_end": True
        }

    state = get_session(request.session_id)

    if not state:
        return {
            "patient_reply": "Session expired.",
            "conversation_end": True
        }

    if state["conversation_end"]:
        return {
            "session_id": request.session_id,
            "patient_reply": "[Consultation already completed politely.]",
            "conversation_end": True
        }

    # Add doctor prescription
    state["messages"].append(HumanMessage(content=request.prescription))

    result = treatment_graph.invoke({
        "messages": state["messages"],
        "clarification_used": False,
        "conversation_end": False
    })

    state["messages"] = result["messages"]
    state["conversation_end"] = result["conversation_end"]

    # Reset session after completion
    if state["conversation_end"]:
        logging.info(f"Session ended: {request.session_id}")
        reset_session(request.session_id)

    return {
        "session_id": request.session_id,
        "patient_reply": state["messages"][-1].content,
        "conversation_end": state["conversation_end"]
    }






