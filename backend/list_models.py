from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get API key from environment
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file")

# Create Groq client
client = Groq(api_key=api_key)

# List all available models for this API key
try:
    models = client.models.list()
    print("Available models for your API key:")
    for model in models:
        print(model)
except Exception as e:
    print(f"Error fetching models: {e}")
