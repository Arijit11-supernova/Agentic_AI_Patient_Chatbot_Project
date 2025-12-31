from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.patient_graph import patient_graph
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'OPTIONS'])
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def handler():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400
    
    user_message = body.get("user_message", "").strip()
    messages = body.get("messages", [])
    
    # First message → greeting
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
    
    # Invoke patient graph
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




