# graph/patient_graph.py
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ------------------ STATE ------------------
class PatientState(TypedDict):
    messages: List[HumanMessage | AIMessage]
    revealed_symptoms: List[str]
    conversation_end: bool

# ------------------ NODE ------------------
def patient_node(state: PatientState):
    messages = state.get("messages", [])
    revealed_symptoms = state.get("revealed_symptoms", [])
    conversation_end = state.get("conversation_end", False)

    # Convert to Groq format
    groq_messages = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            groq_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            groq_messages.append({"role": "assistant", "content": msg.content})

    # Get last doctor message
    last_human_msg = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content.lower()
            break

    # ---------------- SYSTEM PROMPT ----------------
    system_prompt = """You are a simulated patient in a medical consultation.

**Your condition:** You have been experiencing frequent headaches, fatigue, and occasional nausea for the past week.

**Behavior rules:**
- Speak naturally like a real patient, not like a medical textbook
- Keep answers SHORT (1-3 sentences maximum)
- Answer ONLY what the doctor directly asks
- Do NOT volunteer information unless asked
- Do NOT summarize or list all symptoms at once
- Do NOT use bullet points or medical terminology
- Reveal symptoms gradually and naturally
- Show slight concern but remain cooperative
- If asked about duration, say "about a week"
- If asked about severity, describe it conversationally (e.g., "quite bad" or "manageable")

**Examples of good responses:**
- Doctor: "What brings you here today?" → Patient: "I've been having these headaches that just won't go away."
- Doctor: "How long have you had them?" → Patient: "About a week now."
- Doctor: "Any other symptoms?" → Patient: "I've been feeling pretty tired lately, and sometimes a bit nauseous."

**Examples of bad responses (avoid these):**
- Listing symptoms with bullets
- Using medical terms like "experiencing persistent cephalgia"
- Giving too much detail at once
- Summarizing the entire medical history"""

    if conversation_end:
        system_prompt += "\n\nThe consultation is ending. Respond briefly and politely."

    # ---------------- MODEL CALL ----------------
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # ✅ FIXED: Valid Groq model
            messages=[
                {"role": "system", "content": system_prompt},
                *groq_messages
            ],
            temperature=0.7,
            max_tokens=150
        )
        reply = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API Error: {e}")
        reply = "Sorry doctor, could you please repeat that?"

    # ---------------- SYMPTOM TRACKING ----------------
    symptom_keywords = {
        "headache": ["headache", "head hurt", "head pain"],
        "fatigue": ["tired", "fatigue", "exhausted", "no energy"],
        "nausea": ["nausea", "nauseous", "sick", "queasy"]
    }
    
    new_symptoms = []
    reply_lower = reply.lower()
    for symptom, keywords in symptom_keywords.items():
        if symptom not in revealed_symptoms:
            if any(kw in reply_lower for kw in keywords):
                new_symptoms.append(symptom)

    return {
        "messages": messages + [AIMessage(content=reply)],
        "revealed_symptoms": revealed_symptoms + new_symptoms,
        "conversation_end": conversation_end
    }

# ------------------ GRAPH ------------------
builder = StateGraph(PatientState)
builder.add_node("patient", patient_node)
builder.set_entry_point("patient")
builder.add_edge("patient", END)
patient_graph = builder.compile()




