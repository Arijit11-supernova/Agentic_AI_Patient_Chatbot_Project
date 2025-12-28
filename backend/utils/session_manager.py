import uuid

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
    _sessions.pop(session_id, None)
