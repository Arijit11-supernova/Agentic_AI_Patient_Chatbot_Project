from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.evaluator_graph import evaluator_graph

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['POST', 'OPTIONS'])
@app.route('/api/evaluate', methods=['POST', 'OPTIONS'])
def handler():
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







