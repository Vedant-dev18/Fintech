
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=True
)


def analyze_finances(revenue, expenses, profit, margin):
    """Generate an AI explanation of calculated financial results."""

    if not os.getenv("GROQ_API_KEY"):
        return "Groq API key is missing. Check your .env file."

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
    )

    prompt = f"""
    You are FinSight AI, a professional business financial analyst.

    Analyze these business figures:

    Total Revenue: INR {revenue:,.2f}
    Total Expenses: INR {expenses:,.2f}
    Operating Surplus: INR {profit:,.2f}
    Operating Margin: {margin:.2f}%

    Provide:
    1. A simple financial performance summary.
    2. Three practical recommendations to improve profitability.
    3. Two business risks to monitor.

    Important rules:
    - Use only the financial figures provided.
    - Do not invent transactions, trends, or expense categories.
    - Clearly identify recommendations as suggestions.
    - Keep the explanation understandable for a business owner.
    """

    response = llm.invoke(prompt)
    return response.content
