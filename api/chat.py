# api/chat.py

import json
from graph.patient_graph import patient_graph
from langchain_core.messages import HumanMessage, AIMessage

def handler(request):
    # ✅ Only POST allowed
    if request.method != "POST":
        return (
            json.dumps({"error": "Only POST method allowed"}),
            405,
            {"Content-Type": "application/json"}
        )

    try:
        body = request.get_json()
    except Exception:
        return (
            json.dumps({"error": "Invalid JSON body"}),
            400,
            {"Content-Type": "application/json"}
        )

    user_message = body.get("user_message")
    messages = body.get("messages", [])

    # ✅ First message → greeting
    if not user_message and not messages:
        greeting = "Hello! I’m ready to discuss my symptoms with you."
        return (
            json.dumps({
                "reply": greeting,
                "conversation_end": False,
                "messages": [greeting]
            }),
            200,
            {"Content-Type": "application/json"}
        )

    # Build LangGraph message history
    graph_messages = []
    for m in messages:
        if m["role"] == "doctor":
            graph_messages.append(HumanMessage(content=m["content"]))
        else:
            graph_messages.append(AIMessage(content=m["content"]))

    # Add latest doctor message
    graph_messages.append(HumanMessage(content=user_message))

    # Invoke patient graph (STATELESS)
    new_state = patient_graph.invoke({
        "messages": graph_messages,
        "revealed_symptoms": [],
        "conversation_end": False
    })

    reply = new_state["messages"][-1].content

    # Append patient reply to message history
    messages.append({"role": "doctor", "content": user_message})
    messages.append({"role": "patient", "content": reply})

    return (
        json.dumps({
            "reply": reply,
            "conversation_end": new_state["conversation_end"],
            "messages": messages
        }),
        200,
        {"Content-Type": "application/json"}
    )




