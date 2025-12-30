# api/chat.py
from flask import Request, jsonify
from graph.patient_graph import patient_graph
from langchain_core.messages import HumanMessage, AIMessage

def handler(request: Request):
    """
    Vercel serverless function for patient chat
    """
    # ✅ Only POST allowed
    if request.method != "POST":
        return jsonify({"error": "Only POST method allowed"}), 405

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    user_message = body.get("user_message", "").strip()
    messages = body.get("messages", [])

    # ✅ First message → greeting
    if not user_message and not messages:
        greeting = "Hello! I'm ready to discuss my symptoms with you."
        return jsonify({
            "reply": greeting,
            "conversation_end": False,
            "messages": [{"role": "patient", "content": greeting}]
        }), 200

    # Build LangGraph message history
    graph_messages = []
    for m in messages:
        if m.get("role") == "doctor":
            graph_messages.append(HumanMessage(content=m.get("content", "")))
        else:
            graph_messages.append(AIMessage(content=m.get("content", "")))

    # Add latest doctor message
    if user_message:
        graph_messages.append(HumanMessage(content=user_message))

    # Invoke patient graph (STATELESS)
    try:
        new_state = patient_graph.invoke({
            "messages": graph_messages,
            "revealed_symptoms": [],
            "conversation_end": False
        })
    except Exception as e:
        return jsonify({"error": f"Graph invocation failed: {str(e)}"}), 500

    reply = new_state["messages"][-1].content

    # Append messages to history
    if user_message:
        messages.append({"role": "doctor", "content": user_message})
    messages.append({"role": "patient", "content": reply})

    return jsonify({
        "reply": reply,
        "conversation_end": new_state.get("conversation_end", False),
        "messages": messages
    }), 200



