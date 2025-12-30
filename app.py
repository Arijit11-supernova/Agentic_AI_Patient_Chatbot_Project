from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS  # ✅ Add this
import os

from api.chat import handler as chat_handler
from api.evaluate import handler as evaluate_handler
from api.treatment import handler as treatment_handler

app = Flask(__name__, static_folder="frontend")
CORS(app)  # ✅ Allow cross-origin requests for local testing

# Serve index.html at root
@app.route("/")
def home():
    return send_from_directory(app.static_folder, "index.html")

# Route for /api/chat
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        return chat_handler(request)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for /api/evaluate
@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    try:
        return evaluate_handler(request)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route for /api/treatment
@app.route("/api/treatment", methods=["POST"])
def treatment():
    try:
        return treatment_handler(request)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # ✅ Explicitly set host for local testing
    app.run(debug=True, port=5000, host="127.0.0.1")
