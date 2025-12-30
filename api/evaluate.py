from flask import Request, jsonify
import logging

from utils.session_manager import create_session, get_session
from graph.patient_graph import patient_graph
from langchain_core.messages import HumanMessage, AIMessage

logging.basicConfig(level=logging.INFO)

def handler(request: Request):
    if request.method != "POST":
        return jsonify({"error": "Only POST method allowed"}), 405

    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_message = data.get("user_message")
    session_id = data.get("session_id")

    # Create session if missing
    if not session_id:
        session_id = create_session()
        state = get_session(session_id)

        greeting = "New evaluation session started. Ask your question."
        state["messages"].append(AIMessage(content=greeting))
        return jsonify({
            "session_id": session_id,
            "reply": greeting,
            "conversation_end": state["conversation_end"]
        })

    state = get_session(session_id)
    if not state:
        session_id = create_session()
        state = get_session(session_id)
        greeting = "New evaluation session started. Ask your question."
        state["messages"].append(AIMessage(content=greeting))
        return jsonify({
            "session_id": session_id,
            "reply": greeting,
            "conversation_end": state["conversation_end"]
        })

    # Add user message
    state["messages"].append(HumanMessage(content=user_message))

    # Invoke patient graph for evaluation logic
    new_state = patient_graph.invoke({
        "messages": state["messages"],
        "revealed_symptoms": state["revealed_symptoms"],
        "conversation_end": state["conversation_end"]
    })

    # Update session
    state["messages"] = new_state["messages"]
    state["revealed_symptoms"] = new_state["revealed_symptoms"]
    state["conversation_end"] = new_state["conversation_end"]

    return jsonify({
        "session_id": session_id,
        "reply": state["messages"][-1].content,
        "conversation_end": state["conversation_end"]
    })


