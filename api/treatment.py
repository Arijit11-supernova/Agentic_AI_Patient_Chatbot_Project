from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.treatment_graph import treatment_graph
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'OPTIONS'])
@app.route('/api/treatment', methods=['POST', 'OPTIONS'])
def handler():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400
    
    prescription = body.get("prescription", "").strip()
    messages = body.get("messages", [])
    clarification_used = body.get("clarification_used", False)
    
    # First call → greeting
    if not prescription and not messages:
        greeting = "Please share the prescription given by the doctor."
        return jsonify({
            "patient_reply": greeting,
            "conversation_end": False,
            "clarification_used": False,
            "messages": [{"role": "patient", "content": greeting}]
        }), 200
    
    # Build LangGraph messages
    graph_messages = []
    for m in messages:
        if m.get("role") == "doctor":
            graph_messages.append(HumanMessage(content=m.get("content", "")))
        else:
            graph_messages.append(AIMessage(content=m.get("content", "")))
    
    if prescription:
        graph_messages.append(HumanMessage(content=prescription))
    
    try:
        new_state = treatment_graph.invoke({
            "messages": graph_messages,
            "clarification_used": clarification_used,
            "conversation_end": False
        })
    except Exception as e:
        return jsonify({"error": f"Treatment graph error: {str(e)}"}), 500
    
    reply = new_state["messages"][-1].content
    
    if prescription:
        messages.append({"role": "doctor", "content": prescription})
    messages.append({"role": "patient", "content": reply})
    
    return jsonify({
        "patient_reply": reply,
        "conversation_end": new_state.get("conversation_end", False),
        "clarification_used": new_state.get("clarification_used", False),
        "messages": messages
    }), 200








