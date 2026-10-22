
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from financial_tools import calculate_financial_metrics, calculate_savings


# Load the Groq API key securely
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=True
)


# Shared data passed between LangGraph nodes
class FinancialState(TypedDict):
    revenue: float
    expenses: float
    profit: float
    margin: float
    insights: str
    recommendations: str
    savings_5_percent: float
    new_expenses_5_percent: float
    financial_condition: str
    analysis_focus: str


# NODE 1: Calculate verified financial metrics
def calculate_finances(state: FinancialState):
    metrics = calculate_financial_metrics(
        state["revenue"],
        state["expenses"]
    )

    savings = calculate_savings(
        state["expenses"],
        5
    )

    return {
        "profit": metrics["profit"],
        "margin": metrics["margin"],
        "savings_5_percent": savings["savings"],
        "new_expenses_5_percent": savings["new_expenses"]
    }


# Decide which financial analysis path to follow
def choose_financial_path(state: FinancialState):
    margin = state["margin"]

    if margin < 0:
        return "loss"
    elif margin < 15:
        return "low_margin"
    else:
        return "profitable"


# NODE 2A: Loss-making business
def analyze_loss(state: FinancialState):
    return {
        "financial_condition": "Loss-making business",
        "analysis_focus": (
            "Prioritize expense control, monitoring cash flow, "
            "and identifying potential causes of losses."
        )
    }


# NODE 2B: Low-margin business
def analyze_low_margin(state: FinancialState):
    return {
        "financial_condition": "Low-margin business",
        "analysis_focus": (
            "Prioritize cost efficiency, pricing strategy, "
            "and opportunities to improve operating margin."
        )
    }


# NODE 2C: Profitable business
def analyze_profitable(state: FinancialState):
    return {
        "financial_condition": "Profitable business",
        "analysis_focus": (
            "Prioritize sustainable growth, maintaining profitability, "
            "and monitoring potential business risks."
        )
    }


# Configure the Groq AI model
def get_llm():
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY is missing from your .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


# NODE 3: Generate AI financial insights
def generate_insights(state: FinancialState):
    llm = get_llm()

    prompt = f"""
You are FinSight AI, a professional business financial analyst.

Business condition: {state['financial_condition']}
Analysis priority: {state['analysis_focus']}

VERIFIED FINANCIAL DATA:
Revenue: INR {state['revenue']:,.2f}
Expenses: INR {state['expenses']:,.2f}
Operating Surplus/Loss: INR {state['profit']:,.2f}
Operating Margin: {state['margin']:.2f}%

VERIFIED 5% EXPENSE REDUCTION SCENARIO:
Exact savings: INR {state['savings_5_percent']:,.2f}
New expenses: INR {state['new_expenses_5_percent']:,.2f}

TASK:
Explain the company's financial performance in simple language.

Focus on the selected business condition and analysis priority.

IMPORTANT RULES:
1. Use the verified financial figures provided above.
2. Do not invent transactions, trends, or expense categories.
3. Do not calculate hypothetical percentage savings yourself.
4. When mentioning a 5% expense reduction, use the verified figures.
5. Avoid numerical estimates for other reduction percentages.
6. Clearly distinguish operating profit or loss from actual cash flow.
7. Do not claim that hypothetical business risks are proven facts.
8. Keep the explanation understandable for a business owner.
"""

    response = llm.invoke(prompt)

    return {
        "insights": str(response.content)
    }


# NODE 4: Generate business recommendations
def generate_recommendations(state: FinancialState):
    llm = get_llm()

    prompt = f"""
You are FinSight AI, a professional business financial consultant.

BUSINESS CONDITION:
{state['financial_condition']}

ANALYSIS PRIORITY:
{state['analysis_focus']}

VERIFIED FINANCIAL DATA:
Revenue: INR {state['revenue']:,.2f}
Expenses: INR {state['expenses']:,.2f}
Operating Surplus/Loss: INR {state['profit']:,.2f}
Operating Margin: {state['margin']:.2f}%

VERIFIED PYTHON CALCULATION:
If operating expenses are reduced by 5%:
Exact savings: INR {state['savings_5_percent']:,.2f}
New operating expenses: INR {state['new_expenses_5_percent']:,.2f}

PREVIOUS FINANCIAL INSIGHTS:
{state['insights']}

YOUR TASK:
Provide:
1. Three practical business recommendations.
2. Two potential financial risks.

IMPORTANT RULES:
1. Focus on the business condition and analysis priority.
2. Use verified Python calculations for the 5% reduction scenario.
3. Do not recalculate or alter the verified figures.
4. Do not calculate hypothetical savings for other percentages.
5. Do not invent financial facts or unsupported expense categories.
6. Clearly label recommendations as suggestions.
7. Describe potential risks as possibilities, not established facts.
8. Distinguish operating results from actual cash flow.
9. Keep the response clear, practical, and professional.
"""

    response = llm.invoke(prompt)

    return {
        "recommendations": str(response.content)
    }


# BUILD THE LANGGRAPH WORKFLOW
workflow = StateGraph(FinancialState)

# Register all workflow nodes
workflow.add_node("calculate", calculate_finances)

workflow.add_node("loss", analyze_loss)
workflow.add_node("low_margin", analyze_low_margin)
workflow.add_node("profitable", analyze_profitable)

workflow.add_node("insights", generate_insights)
workflow.add_node("recommendations", generate_recommendations)

# Start with financial calculations
workflow.add_edge(START, "calculate")

# Conditional routing based on operating margin
workflow.add_conditional_edges(
    "calculate",
    choose_financial_path,
    {
        "loss": "loss",
        "low_margin": "low_margin",
        "profitable": "profitable"
    }
)

# All financial paths lead to AI insights
workflow.add_edge("loss", "insights")
workflow.add_edge("low_margin", "insights")
workflow.add_edge("profitable", "insights")

# Generate recommendations after insights
workflow.add_edge("insights", "recommendations")

# Finish the workflow
workflow.add_edge("recommendations", END)

# Compile the workflow for use in app.py
financial_agent = workflow.compile()
