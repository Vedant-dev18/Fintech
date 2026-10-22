
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

api_key = os.getenv("GROQ_API_KEY")
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

print("Connecting to Groq AI...")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key
)

response = llm.invoke(
    "Explain what business profit means in two simple sentences."
)

print("\nGroq AI Response:")
print(response.content)
