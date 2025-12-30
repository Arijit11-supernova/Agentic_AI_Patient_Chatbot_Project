# graph/treatment_graph.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------ STATE ------------------
class TreatmentState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    clarification_used: bool
    conversation_end: bool

# ------------------ NODE ------------------
def treatment_node(state: TreatmentState):
    messages = state.get("messages", [])
    clarification_used = state.get("clarification_used", False)
    conversation_end = state.get("conversation_end", False)

    if conversation_end:
        return state

    # Convert to Groq format
    groq_messages = []
    for msg in messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        groq_messages.append({"role": role, "content": msg.content})

    # ---------------- SYSTEM PROMPT ----------------
    if not clarification_used:
        system_prompt = """You are a patient receiving a prescription from a doctor.

**Instructions:**
- Read the doctor's prescription carefully
- If ANYTHING is unclear (dosage, timing, duration, how to take it), ask ONE clarifying question
- Be natural and conversational
- Do NOT accept the prescription yet - first clarify if needed
- If everything is clear, say you understand and will follow it

**Examples:**
- Doctor: "Take this paracetamol twice daily" → Patient: "Thank you doctor. Should I take it before or after meals?"
- Doctor: "Apply this cream on the affected area" → Patient: "How many times a day should I apply it?"
- Doctor: "Take 500mg of amoxicillin three times a day for 5 days, preferably after meals" → Patient: "Thank you doctor, I understand. I'll take it as prescribed."
"""
    else:
        system_prompt = """You are a patient whose question has been answered by the doctor.

**Instructions:**
- Thank the doctor politely
- Confirm you understand and will follow the prescription
- Keep it brief (1-2 sentences)
- End the conversation naturally

**Example:**
"Thank you doctor, that's clear now. I'll make sure to take it as you've explained."
"""

    # ---------------- MODEL CALL ----------------
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ FIXED: Valid Groq model
            messages=[{"role": "system", "content": system_prompt}, *groq_messages],
            temperature=0.7,
            max_tokens=100
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API Error: {e}")
        reply = "Thank you doctor, I understand and will follow your advice."
        clarification_used = True
        conversation_end = True

    # ---------------- CLARIFICATION LOGIC ----------------
    clarification_indicators = ["?", "how", "when", "should i", "do i", "which", "what"]
    asked_question = any(indicator in reply.lower() for indicator in clarification_indicators)

    if clarification_used:
        # Already asked a question, now accept
        conversation_end = True
    elif asked_question:
        # First clarification
        clarification_used = True
        conversation_end = False
    else:
        # Accepted directly without questions
        conversation_end = True

    return {
        "messages": messages + [AIMessage(content=reply)],
        "clarification_used": clarification_used,
        "conversation_end": conversation_end
    }

# ------------------ GRAPH ------------------
builder = StateGraph(TreatmentState)
builder.add_node("treatment", treatment_node)
builder.set_entry_point("treatment")
builder.add_edge("treatment", END)
treatment_graph = builder.compile()




