# api/chat.py

import json
from utils.session_manager import create_session, get_session
from graph.patient_graph import patient_graph
from langchain_core.messages import HumanMessage, AIMessage

def handler(request):
    # Only POST allowed
    if request["method"] != "POST":
        return {
            "statusCode": 405,
            "body": json.dumps({"error": "Only POST method allowed"})
        }

    try:
        data = json.loads(request["body"])
    except Exception:
        return {
            "statusCode": 400,
            "body": json.dumps({"error": "Invalid JSON body"})
        }

    user_message = data.get("user_message")
    session_id = data.get("session_id")

    # If no session_id, create a new session
    if not session_id:
        session_id = create_session()
        state = get_session(session_id)
        greeting = "Hello! I’m ready to discuss my symptoms with you."
        state["messages"].append(AIMessage(content=greeting))
        return {
            "statusCode": 200,
            "body": json.dumps({
                "session_id": session_id,
                "reply": greeting,
                "conversation_end": state["conversation_end"]
            })
        }

    # Existing session
    state = get_session(session_id)
    if not state:
        session_id = create_session()
        state = get_session(session_id)
        greeting = "Hello! I’m ready to discuss my symptoms with you."
        state["messages"].append(AIMessage(content=greeting))
        return {
            "statusCode": 200,
            "body": json.dumps({
                "session_id": session_id,
                "reply": greeting,
                "conversation_end": state["conversation_end"]
            })
        }

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

    return {
        "statusCode": 200,
        "body": json.dumps({
            "session_id": session_id,
            "reply": state["messages"][-1].content,
            "conversation_end": state["conversation_end"]
        })
    }


