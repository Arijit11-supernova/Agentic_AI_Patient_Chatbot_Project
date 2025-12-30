from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class TreatmentState(TypedDict):
    messages: List
    clarification_used: bool
    conversation_end: bool


def treatment_node(state: TreatmentState):
    if state["conversation_end"]:
        return state

    groq_messages = []
    for msg in state["messages"]:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        groq_messages.append({"role": role, "content": msg.content})

    system_prompt = """
You are a patient responding to a doctor's prescription.

Rules:
- Respond naturally
- Ask at most ONE clarification if confused
- If clarification already used, politely accept
- Do NOT give medical advice
"""

    response = client.chat.completions.create(
        model="groq/compound",
        messages=[{"role": "system", "content": system_prompt}, *groq_messages]
    )

    reply = response.choices[0].message.content.strip()

    clarification_triggers = [
        "could you explain",
        "can you clarify",
        "what does that mean"
    ]

    asked_clarification = any(t in reply.lower() for t in clarification_triggers)

    clarification_used = state["clarification_used"]
    conversation_end = True

    if clarification_used:
        reply = "Thank you, doctor. I understand and will follow your advice."

    elif asked_clarification:
        clarification_used = True

    return {
        "messages": state["messages"] + [AIMessage(content=reply)],
        "clarification_used": clarification_used,
        "conversation_end": conversation_end
    }


builder = StateGraph(TreatmentState)
builder.add_node("treatment", treatment_node)
builder.set_entry_point("treatment")
builder.add_edge("treatment", END)

treatment_graph = builder.compile()

