
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pathlib import Path

from financial_tools import calculate_savings

load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=True
)


@tool
def savings_calculator(
    expenses: float,
    reduction_percent: float
) -> dict:
    """Calculate exact savings and new expenses after a percentage reduction."""
    return calculate_savings(expenses, reduction_percent)


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

llm_with_tools = llm.bind_tools([savings_calculator])

response = llm_with_tools.invoke(
    "Our business expenses are INR 70000. "
    "Use the savings_calculator tool to calculate "
    "the savings from a 12 percent expense reduction."
)

if response.tool_calls:
    print("AI REQUESTED A TOOL:")
    for call in response.tool_calls:
        print("Tool:", call["name"])
        print("Arguments:", call["args"])

        result = savings_calculator.invoke(call["args"])
        print("VERIFIED TOOL RESULT:", result)
        
        from langchain_core.messages import ToolMessage

        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=call["id"]
        )

        final_response = llm_with_tools.invoke([
            (
                "system",
                "You are FinSight AI. Explain financial tool results "
                "clearly and use only verified numbers."
            ),
            (
                "human",
                "Our business expenses are INR 70000. "
                "What happens if we reduce them by 12%?"
            ),
            response,
            tool_message
        ])

        print("\nFINAL AI ANSWER:")
        print(final_response.content)

else:
    print("AI did not request a tool.")
    print("Response:", response.content)
