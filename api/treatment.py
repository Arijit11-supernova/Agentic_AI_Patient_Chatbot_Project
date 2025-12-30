# api/treatment.py

import json
from graph.treatment_graph import treatment_graph
from langchain_core.messages import HumanMessage, AIMessage

def handler(request):
    # ✅ Allow only POST
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

    prescription = body.get("prescription")
    messages = body.get("messages", [])
    clarification_used = body.get("clarification_used", False)

    # First call → greeting
    if not prescription and not messages:
        greeting = "Please share the prescription given by the doctor."
        return (
            json.dumps({
                "patient_reply": greeting,
                "conversation_end": False,
                "clarification_used": False,
                "messages": [
                    {"role": "patient", "content": greeting}
                ]
            }),
            200,
            {"Content-Type": "application/json"}
        )

    # Build LangGraph messages
    graph_messages = []
    for m in messages:
        if m["role"] == "doctor":
            graph_messages.append(HumanMessage(content=m["content"]))
        else:
            graph_messages.append(AIMessage(content=m["content"]))

    # Add prescription as doctor message
    graph_messages.append(HumanMessage(content=prescription))

    # Invoke treatment graph (STATELESS)
    new_state = treatment_graph.invoke({
        "messages": graph_messages,
        "clarification_used": clarification_used,
        "conversation_end": False
    })

    reply = new_state["messages"][-1].content

    # Update message history
    messages.append({"role": "doctor", "content": prescription})
    messages.append({"role": "patient", "content": reply})

    return (
        json.dumps({
            "patient_reply": reply,
            "conversation_end": new_state["conversation_end"],
            "clarification_used": new_state["clarification_used"],
            "messages": messages
        }),
        200,
        {"Content-Type": "application/json"}
    )





