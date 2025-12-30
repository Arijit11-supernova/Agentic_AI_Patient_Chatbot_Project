# api/evaluate.py

import json
from graph.evaluator_graph import evaluator_graph

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

    doctor_message = body.get("doctor_message")
    patient_history = body.get("patient_history", [])

    if not doctor_message:
        return (
            json.dumps({"error": "doctor_message is required"}),
            400,
            {"Content-Type": "application/json"}
        )

    # ✅ Invoke evaluator graph (stateless)
    result = evaluator_graph.invoke({
        "doctor_message": doctor_message,
        "patient_history": patient_history,
        "evaluation": {}
    })

    return (
        json.dumps({
            "evaluation": result["evaluation"]
        }),
        200,
        {"Content-Type": "application/json"}
    )





