"""
core/agent.py
─────────────
LangGraph ReAct agent powered by Groq (llama-3.3-70b-versatile).
Tools wrap engine functions so the agent can fetch real data.
"""

import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from core.engine import (
    get_fundamental_data,
    get_company_info,
    calculate_basic_risk_metrics,
    analyze_stock,
    get_news,
)

# ─────────────────────────────────────────
# TOOL DEFINITIONS
# Only wrap functions that return text/dict —
# NOT chart functions (they return Plotly objects
# which can't be serialized as tool responses).
# ─────────────────────────────────────────

tools = [
    StructuredTool.from_function(
        name="analyze_stock",
        func=analyze_stock,
        description=(
            "Get a trend and risk summary for a stock. "
            "Args: ticker (str), period (str, e.g. '1y', '6mo')."
        ),
    ),
    StructuredTool.from_function(
        name="get_fundamental_data",
        func=get_fundamental_data,
        description=(
            "Get financial fundamentals: P/E, EPS, market cap, revenue, etc. "
            "Args: ticker (str)."
        ),
    ),
    StructuredTool.from_function(
        name="get_company_info",
        func=get_company_info,
        description=(
            "Get company name, sector, and business description. "
            "Args: ticker (str)."
        ),
    ),
    StructuredTool.from_function(
        name="calculate_basic_risk_metrics",
        func=calculate_basic_risk_metrics,
        description=(
            "Calculate volatility, Sharpe Ratio, and Max Drawdown. "
            "Args: ticker (str), period (str, e.g. '1y')."
        ),
    ),
    StructuredTool.from_function(
        name="get_news",
        func=get_news,
        description=(
            "Fetch the latest financial news for a stock. "
            "Args: ticker (str)."
        ),
    ),
]

_SYSTEM_PROMPT = """You are an expert financial analyst AI assistant.
Use the available tools to fetch real data when answering questions about stocks.
Always provide structured, concise, and professional analysis.
Never make up stock prices or financial data — always use the tools.
When you have the data, synthesize it into a clear, actionable response.
Include trend, risk level, and key metrics whenever relevant.
"""


class _AgentAdapter:
    """
    Wraps the LangGraph agent so app.py can call:
        agent.invoke({"input": "..."})
    and get back:
        {"output": "..."}
    """
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("input", "")
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ]
        try:
            response = self.graph.invoke(
                {"messages": messages},
                {"recursion_limit": 12},
            )
            final = response["messages"][-1]
            return {"output": final.content}
        except Exception as e:
            return {"output": f"Agent error: {e}"}


@st.cache_resource
def create_agent_executor():
    """
    Creates and caches the LangGraph agent for the session.
    Reads GROQ_API_KEY from Streamlit secrets or environment variable.
    """
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to .streamlit/secrets.toml or as a Render environment variable."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=api_key,
    )

    graph = create_react_agent(llm, tools=tools)
    return _AgentAdapter(graph)