from agent_workflow import financial_agent
import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="FinSight AI", page_icon="📊", layout="wide")
DB_PATH = Path(__file__).resolve().parent / "finsight.db"
REQUIRED = ["Date", "Category", "Amount", "Type"]


def connect():
    return sqlite3.connect(DB_PATH)


def init_database():
    with connect() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            amount REAL NOT NULL CHECK(amount >= 0),
            type TEXT NOT NULL CHECK(type IN ('Revenue', 'Expense'))
        )""")


def validate_csv(file):
    try:
        df = pd.read_csv(file)
    except Exception as exc:
        raise ValueError(f"Could not read CSV: {exc}") from exc
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    if df.empty:
        raise ValueError("The CSV has no transactions.")
    df = df[REQUIRED].copy()
    dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    amounts = pd.to_numeric(df["Amount"], errors="coerce")
    types = df["Type"].astype("string").str.strip().str.title()
    categories = df["Category"].astype("string").str.strip()
    invalid = (dates.isna() | amounts.isna() | (amounts < 0)
               | ~types.isin(["Revenue", "Expense"]).fillna(False)
               | categories.isna() | (categories == ""))
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} invalid row(s). Dates must be YYYY-MM-DD; amounts must be non-negative; types must be Revenue or Expense.")
    df["Date"] = dates.dt.strftime("%Y-%m-%d")
    df["Amount"] = amounts.astype(float)
    df["Type"] = types.astype(str)
    df["Category"] = categories.astype(str)
    return df


def save_replace(df):
    records = list(df[REQUIRED].itertuples(index=False, name=None))
    with connect() as conn:
        conn.execute("DELETE FROM transactions")
        conn.executemany("INSERT INTO transactions (date, category, amount, type) VALUES (?, ?, ?, ?)", records)


def load_transactions():
    with connect() as conn:
        return pd.read_sql_query(
            "SELECT date AS Date, category AS Category, amount AS Amount, type AS Type FROM transactions ORDER BY date, id", conn
        )


init_database()
st.title("📊 FinSight AI")
st.subheader("Business Financial Analyst")
st.caption("Step 9: Financial dashboard with persistent SQLite storage")

st.sidebar.header("📁 Financial Data")
st.sidebar.success("SQLite database connected")
uploaded = st.sidebar.file_uploader("Upload financial CSV", type=["csv"])
st.sidebar.caption("Required columns: Date, Category, Amount, Type")

if uploaded is not None:
    try:
        incoming = validate_csv(uploaded)
        st.sidebar.info(f"File validated: {len(incoming)} transactions. Saving replaces all previously saved records.")
        if st.sidebar.button("💾 Save to Database", type="primary"):
            save_replace(incoming)
            st.sidebar.success(f"Saved {len(incoming)} transactions!")
            st.rerun()
    except ValueError as exc:
        st.sidebar.error(str(exc))

df = load_transactions()
if df.empty:
    st.info("No records saved yet. Upload your CSV in the sidebar, then click 'Save to Database'.")
    st.stop()

st.caption(f"Loaded {len(df)} saved transactions from finsight.db")
df["Date"] = pd.to_datetime(df["Date"])
df["Month"] = df["Date"].dt.strftime("%Y-%m")
months = sorted(df["Month"].unique().tolist())
selected = st.sidebar.multiselect("Months to analyze", months, default=months)
if not selected:
    st.warning("Select at least one month.")
    st.stop()
view = df[df["Month"].isin(selected)].copy()
revenue = float(view.loc[view["Type"] == "Revenue", "Amount"].sum())
expenses = float(view.loc[view["Type"] == "Expense", "Amount"].sum())
surplus = revenue - expenses
margin = surplus / revenue * 100 if revenue else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"₹{revenue:,.0f}")
c2.metric("Total Expenses", f"₹{expenses:,.0f}")
c3.metric("Operating Surplus", f"₹{surplus:,.0f}")
c4.metric("Surplus Margin", f"{margin:.1f}%" if margin is not None else "N/A")

st.divider()
st.subheader("📈 Monthly Revenue and Expenses")
monthly = view.groupby(["Month", "Type"], as_index=False)["Amount"].sum()
st.plotly_chart(px.bar(monthly, x="Month", y="Amount", color="Type", barmode="group"), use_container_width=True)

st.subheader("🍩 Expense Breakdown")
expense_rows = view[view["Type"] == "Expense"]
if expense_rows.empty:
    st.info("No expenses for the selected period.")
else:
    grouped = expense_rows.groupby("Category", as_index=False)["Amount"].sum()
    st.plotly_chart(px.pie(grouped, names="Category", values="Amount", hole=0.4), use_container_width=True)

st.subheader("📋 Saved Business Transactions")
st.dataframe(view.drop(columns="Month"), use_container_width=True, hide_index=True)
st.subheader("💡 Financial Summary")
st.info(f"Revenue ₹{revenue:,.0f}; expenses ₹{expenses:,.0f}; operating surplus ₹{surplus:,.0f}; "
        + (f"surplus margin {margin:.1f}%." if margin is not None else "surplus margin unavailable (no revenue)."))
st.caption("Figures reflect recorded transactions only; AI is not connected yet.")

# -------- FIN SIGHT AI ANALYST --------

st.divider()
st.subheader("🤖 FinSight AI Financial Analyst")

st.write(
    "Generate AI-powered financial insights and "
    "business recommendations using Groq."
)


if st.button("🧠 Analyze My Business with AI"):
    with st.spinner("LangGraph agents are analyzing your business..."):
        try:
            result = financial_agent.invoke({
                "revenue": float(revenue),
                "expenses": float(expenses),
                "profit": 0.0,
                "margin": 0.0,
                "insights": "",
                "recommendations": ""
            })

            st.session_state["ai_analysis"] = (
                "### 📊 Financial Insights\n\n"
                + result["insights"]
                + "\n\n### 💡 Business Recommendations\n\n"
                + result["recommendations"]
            )

            st.success("LangGraph analysis completed successfully!")

        except Exception as error:
            st.error(f"LangGraph analysis failed: {error}")

if "ai_analysis" in st.session_state:
    st.markdown(st.session_state["ai_analysis"])
