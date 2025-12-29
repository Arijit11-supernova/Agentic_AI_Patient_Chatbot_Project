from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import logging

from utils.session_manager import create_session, get_session, reset_session
from graph.patient_graph import patient_graph
from graph.evaluator_graph import evaluator_graph
from graph.treatment_graph import treatment_graph
from langchain_core.messages import HumanMessage

load_dotenv()
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# ================= ROOT =================
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Agentic AI Patient Chatbot Backend Running"})

# ================= CHAT =================
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    query = data.get("query")
    session_id = data.get("session_id")

    # Create or fetch session
    if not session_id:
        session_id = create_session()
        state = get_session(session_id)
        logging.info(f"New session created: {session_id}")
    else:
        state = get_session(session_id)
        if not state:
            session_id = create_session()
            state = get_session(session_id)

    if state["conversation_end"]:
        return jsonify({
            "session_id": session_id,
            "reply": "[Conversation already completed politely.]",
            "conversation_end": True
        })

    # Append doctor message
    state["messages"].append(HumanMessage(content=query))

    # Invoke patient graph
    new_state = patient_graph.invoke({
        "messages": state["messages"],
        "revealed_symptoms": state["revealed_symptoms"],
        "conversation_end": False
    })

    state["messages"] = new_state["messages"]
    state["revealed_symptoms"] = new_state["revealed_symptoms"]
    state["conversation_end"] = new_state["conversation_end"]

    return jsonify({
        "session_id": session_id,
        "reply": state["messages"][-1].content,
        "conversation_end": state["conversation_end"]
    })

# ================= EVALUATION =================
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    result = evaluator_graph.invoke({
        "doctor_message": data["doctor_message"],
        "patient_history": data["patient_history"]
    })
    return jsonify({"evaluation": result})

# ================= TREATMENT =================
@app.route("/treatment", methods=["POST"])
def treatment():
    data = request.get_json()
    prescription = data.get("prescription")
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({
            "patient_reply": "No active session found.",
            "conversation_end": True
        })

    state = get_session(session_id)

    if not state:
        return jsonify({
            "patient_reply": "Session expired.",
            "conversation_end": True
        })

    if state["conversation_end"]:
        return jsonify({
            "session_id": session_id,
            "patient_reply": "[Consultation already completed politely.]",
            "conversation_end": True
        })

    # Add doctor prescription
    state["messages"].append(HumanMessage(content=prescription))

    result = treatment_graph.invoke({
        "messages": state["messages"],
        "clarification_used": False,
        "conversation_end": False
    })

    state["messages"] = result["messages"]
    state["conversation_end"] = result["conversation_end"]

    # Reset session after completion
    if state["conversation_end"]:
        logging.info(f"Session ended: {session_id}")
        reset_session(session_id)

    return jsonify({
        "session_id": session_id,
        "patient_reply": state["messages"][-1].content,
        "conversation_end": state["conversation_end"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)





