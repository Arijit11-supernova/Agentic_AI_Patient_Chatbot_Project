from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
import os

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="groq/compound"  
)

patient_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a virtual patient.
            You describe symptoms slowly and naturally.
            Do NOT reveal everything at once.
            Answer only what the doctor asks.
            If treatment is correct, accept it politely.
            If incorrect, express concern."""
        ),
        ("human", "{input}")
    ]
)

def patient_response(user_input: str, state: dict):
    try:
        chain = patient_prompt | llm
        response = chain.invoke({"input": user_input})
        return response.content
    except Exception as e:
        return f"[Error generating response: {e}]"

