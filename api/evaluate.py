# api/evaluate.py
from flask import Request, jsonify
from graph.evaluator_graph import evaluator_graph

def handler(request: Request):
    """Vercel serverless function for evaluation"""
    if request.method != "POST":
        return jsonify({"error": "Only POST method allowed"}), 405

    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    doctor_message = body.get("doctor_message", "").strip()
    patient_history = body.get("patient_history", [])

    if not doctor_message:
        return jsonify({"error": "doctor_message is required"}), 400

    # Invoke evaluator graph
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






