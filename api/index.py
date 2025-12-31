from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.patient_graph import patient_graph
from graph.evaluator_graph import evaluator_graph
from graph.treatment_graph import treatment_graph
from langchain_core.messages import HumanMessage, AIMessage

app = Flask(__name__)
CORS(app)

# ==================== CHAT ENDPOINT ====================
@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
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
    
    if user_message:
        graph_messages.append(HumanMessage(content=user_message))
    
    try:
        new_state = patient_graph.invoke({
            "messages": graph_messages,
            "revealed_symptoms": [],
            "conversation_end": False
        })
    except Exception as e:
        return jsonify({"error": f"Graph invocation failed: {str(e)}"}), 500
    
    reply = new_state["messages"][-1].content
    
    if user_message:
        messages.append({"role": "doctor", "content": user_message})
    messages.append({"role": "patient", "content": reply})
    
    return jsonify({
        "reply": reply,
        "conversation_end": new_state.get("conversation_end", False),
        "messages": messages
    }), 200

# ==================== EVALUATE ENDPOINT ====================
@app.route('/api/evaluate', methods=['POST', 'OPTIONS'])
def evaluate():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400
    
    doctor_message = body.get("doctor_message", "").strip()
    patient_history = body.get("patient_history", [])
    
    if not doctor_message:
        return jsonify({"error": "doctor_message is required"}), 400
    
    try:
        result = evaluator_graph.invoke({
            "doctor_message": doctor_message,
            "patient_history": patient_history,
            "evaluation": {}
        })
    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500
    
    return jsonify({
        "evaluation": result.get("evaluation", {})
    }), 200

# ==================== TREATMENT ENDPOINT ====================
@app.route('/api/treatment', methods=['POST', 'OPTIONS'])
def treatment():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400
    
    prescription = body.get("prescription", "").strip()
    messages = body.get("messages", [])
    clarification_used = body.get("clarification_used", False)
    
    if not prescription and not messages:
        greeting = "Please share the prescription given by the doctor."
        return jsonify({
            "patient_reply": greeting,
            "conversation_end": False,
            "clarification_used": False,
            "messages": [{"role": "patient", "content": greeting}]
        }), 200
    
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
