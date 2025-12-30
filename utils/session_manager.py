import uuid
from langchain_core.messages import HumanMessage, AIMessage

# In-memory session store
# NOTE: Vercel serverless instances are ephemeral,
# but this is ACCEPTABLE for the assignment demo.
_sessions = {}


def create_session():
    session_id = str(uuid.uuid4())

    _sessions[session_id] = {
        "messages": [],
        "revealed_symptoms": [],
        "conversation_end": False
    }

    return session_id


def get_session(session_id):
    return _sessions.get(session_id)


def reset_session(session_id):
    if session_id in _sessions:
        del _sessions[session_id]
