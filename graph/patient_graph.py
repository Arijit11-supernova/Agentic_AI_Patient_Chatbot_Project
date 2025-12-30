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
    messages: List
    revealed_symptoms: List[str]
    conversation_end: bool


# ------------------ NODE ------------------
def patient_node(state: PatientState):
    groq_messages = []

    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            groq_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            groq_messages.append({"role": "assistant", "content": msg.content})

    # Safe defaults
    revealed_symptoms = state.get("revealed_symptoms", [])
    conversation_end = state.get("conversation_end", False)

    # Get last doctor message
    last_human_msg = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human_msg = msg.content.lower()
            break

    # Detect prescription intent
    prescription_keywords = [
        "take", "tablet", "pill", "medicine",
        "medication", "prescribe", "dosage"
    ]

    prescription_given = (
        last_human_msg
        and any(word in last_human_msg for word in prescription_keywords)
    )

    # ---------------- SYSTEM PROMPT ----------------
    system_prompt = (
        "You are a simulated patient in a medical consultation.\n"
        "Behavior rules:\n"
        "- Speak naturally like a real human patient.\n"
        "- Keep answers SHORT and conversational.\n"
        "- Answer ONLY what the doctor asks.\n"
        "- Do NOT summarize past symptoms unless explicitly asked.\n"
        "- Do NOT use bullet points, lists, or medical-style reports.\n"
        "- Reveal symptoms gradually and only when relevant to the question.\n"
        "- Never repeat already revealed symptoms.\n"
        "- Do NOT give medical advice, diagnoses, or treatment suggestions.\n"
    )

    if prescription_given and not conversation_end:
        system_prompt += (
            "\nThe doctor has prescribed a treatment.\n"
            "- Respond politely like a patient.\n"
            "- Ask ONLY ONE simple clarifying question (dosage, timing, or side effects).\n"
            "- After this reply, the conversation must end.\n"
        )
    elif conversation_end:
        system_prompt += (
            "\nThe consultation is complete.\n"
            "- End the conversation politely in one short sentence.\n"
        )

    # ---------------- MODEL CALL ----------------
    response = client.chat.completions.create(
        model="groq/compound",
        messages=[
            {"role": "system", "content": system_prompt},
            *groq_messages
        ]
    )

    reply = response.choices[0].message.content.strip()

    # ---------------- SYMPTOM TRACKING ----------------
    symptom_keywords = ["headache", "tired", "fatigue", "nausea", "dizzy", "cough", "fever"]
    new_symptoms = [
        s for s in symptom_keywords
        if s in reply.lower() and s not in revealed_symptoms
    ]

    # ---------------- END CONVERSATION ----------------
    if prescription_given:
        conversation_end = True

    return {
        "messages": state["messages"] + [AIMessage(content=reply)],
        "revealed_symptoms": revealed_symptoms + new_symptoms,
        "conversation_end": conversation_end
    }


# ------------------ GRAPH ------------------
builder = StateGraph(PatientState)
builder.add_node("patient", patient_node)
builder.set_entry_point("patient")
builder.add_edge("patient", END)

patient_graph = builder.compile()

