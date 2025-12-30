# graph/evaluator_graph.py
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
import os
import json
import logging

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
logging.basicConfig(level=logging.INFO)

class EvaluatorState(TypedDict):
    doctor_message: str
    patient_history: List[str]
    evaluation: Dict

def evaluator_node(state: EvaluatorState):
    """
    Evaluates the doctor's question against patient history using Groq.
    Returns classification: RELEVANT, IRRELEVANT, or REPETITIVE
    """
    history_text = "\n".join(state.get("patient_history", []))
    doctor_msg = state.get("doctor_message", "")

    prompt = f"""You are a medical conversation evaluator. Analyze this doctor's question.

**Patient conversation history:**
{history_text if history_text else "No prior conversation"}

**Doctor's current question:**
"{doctor_msg}"

**Task:** Classify this question into ONE of these categories:

1. **RELEVANT** - The question appropriately follows the conversation and helps diagnose the patient
2. **IRRELEVANT** - The question is off-topic or unrelated to the patient's medical concern
3. **REPETITIVE** - The question asks about something already discussed in detail

**Response format (JSON only):**
{{
  "verdict": "RELEVANT" | "IRRELEVANT" | "REPETITIVE",
  "reason": "Brief explanation (one sentence)",
  "suggestion": "Optional suggestion for improvement (if verdict is not RELEVANT)"
}}

**Examples:**

Example 1:
History: "Patient: I've been having headaches. Doctor: How long? Patient: About a week."
Question: "Are the headaches getting worse?"
Response: {{"verdict": "RELEVANT", "reason": "Asking about symptom progression is medically appropriate"}}

Example 2:
History: "Patient: I have a headache."
Question: "What's your favorite color?"
Response: {{"verdict": "IRRELEVANT", "reason": "Question unrelated to medical concern", "suggestion": "Ask about symptom characteristics or duration"}}

Example 3:
History: "Patient: Headache for a week. Doctor: How long? Patient: I said about a week."
Question: "How long have you had the headache?"
Response: {{"verdict": "REPETITIVE", "reason": "Patient already stated duration twice", "suggestion": "Move to other aspects like severity or triggers"}}

Now evaluate the current question. Respond ONLY with valid JSON, no other text.
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ FIXED: Valid Groq model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        raw_content = response.choices[0].message.content.strip()
        
        # Try to extract JSON from response
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_content:
            raw_content = raw_content.split("```")[1].split("```")[0].strip()
        
        evaluation = json.loads(raw_content)
        
        # Validate structure
        if "verdict" not in evaluation:
            evaluation = {"verdict": "RELEVANT", "reason": "Evaluation parsing failed"}
            
    except json.JSONDecodeError as e:
        logging.warning(f"Invalid JSON from Groq: {raw_content}")
        evaluation = {"verdict": "RELEVANT", "reason": "Could not parse evaluation", "raw": raw_content}
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        evaluation = {"verdict": "ERROR", "reason": str(e)}

    return {
        "doctor_message": state.get("doctor_message", ""),
        "patient_history": state.get("patient_history", []),
        "evaluation": evaluation
    }

# Build the state graph
builder = StateGraph(EvaluatorState)
builder.add_node("evaluator", evaluator_node)
builder.set_entry_point("evaluator")
builder.add_edge("evaluator", END)
evaluator_graph = builder.compile()


