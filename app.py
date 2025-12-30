# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# Import API handlers
from api.chat import handler as chat_handler
from api.evaluate import handler as evaluate_handler
from api.treatment import handler as treatment_handler

app = Flask(__name__, static_folder="frontend")
CORS(app)  # Enable CORS for all routes

# ==================== ROUTES ====================

@app.route("/")
def home():
    """Serve the main HTML page"""
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    """Patient chat endpoint"""
    if request.method == "OPTIONS":
        return "", 200
    try:
        response, status_code = chat_handler(request)
        return response, status_code
    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/evaluate", methods=["POST", "OPTIONS"])
def evaluate():
    """Doctor question evaluation endpoint"""
    if request.method == "OPTIONS":
        return "", 200
    try:
        response, status_code = evaluate_handler(request)
        return response, status_code
    except Exception as e:
        print(f"Error in /api/evaluate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/treatment", methods=["POST", "OPTIONS"])
def treatment():
    """Treatment/prescription endpoint"""
    if request.method == "OPTIONS":
        return "", 200
    try:
        response, status_code = treatment_handler(request)
        return response, status_code
    except Exception as e:
        print(f"Error in /api/treatment: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== MAIN ====================

if __name__ == "__main__":
    print("🚀 Starting Flask server...")
    print("📍 Access app at: http://127.0.0.1:5000")
    app.run(
        debug=True, 
        port=5000, 
        host="127.0.0.1"
    )

