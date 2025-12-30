from flask import Request, jsonify
import logging

from utils.session_manager import create_session, get_session
from graph.patient_graph import patient_graph
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO)

def handler(request: Request):
    # Allow only POST
    if request.method != "POST":
        return jsonify({"error": "Only POST method allowed"}), 405

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_message = data.get("user_message")
    session_id = data.get("session_id")

    # If no session_id, create a new session
    if not session_id:
        session_id = create_session()
        state = get_session(session_id)

        # ✅ Automatically generate first patient greeting
        greeting = "Hello! I’m ready to discuss my symptoms with you."
        state["messages"].append(AIMessage(content=greeting))
        return jsonify({
            "session_id": session_id,
            "reply": greeting,
            "conversation_end": state["conversation_end"]
        })

    # Existing session
    state = get_session(session_id)
    if not state:
        # Expired or invalid session
        session_id = create_session()
        state = get_session(session_id)

        # Generate first patient greeting
        greeting = "Hello! I’m ready to discuss my symptoms with you."
        state["messages"].append(AIMessage(content=greeting))
        return jsonify({
            "session_id": session_id,
            "reply": greeting,
            "conversation_end": state["conversation_end"]
        })

    # Add user message to conversation
    state["messages"].append(HumanMessage(content=user_message))

    # Invoke patient graph
    new_state = patient_graph.invoke({
        "messages": state["messages"],
        "revealed_symptoms": state["revealed_symptoms"],
        "conversation_end": state["conversation_end"]
    })

    # Update session state
    state["messages"] = new_state["messages"]
    state["revealed_symptoms"] = new_state["revealed_symptoms"]
    state["conversation_end"] = new_state["conversation_end"]

    return jsonify({
        "session_id": session_id,
        "reply": state["messages"][-1].content,
        "conversation_end": state["conversation_end"]
    })
