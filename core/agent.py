import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage

from core.engine import (
    get_stock_price_chart,
    get_news,
    get_fundamental_data,
    get_price_forecast,
    get_company_info,
    calculate_basic_risk_metrics
)

# ─────────────────────────────────────────
# TOOLS
# Each tool wraps an engine function so the
# agent can call them during reasoning.
# ─────────────────────────────────────────
tools = [
    StructuredTool.from_function(
        name="get_stock_price_chart",
        func=get_stock_price_chart,
        description="Get historical stock price chart. Args: ticker_symbol (str), period (str)."
    ),
    StructuredTool.from_function(
        name="get_fundamental_data",
        func=get_fundamental_data,
        description="Get financial fundamentals like P/E, EPS, market cap. Args: ticker_symbol (str)."
    ),
    StructuredTool.from_function(
        name="get_price_forecast",
        func=get_price_forecast,
        description="Get 14-day AI price forecast using Prophet. Args: ticker_symbol (str)."
    ),
    StructuredTool.from_function(
        name="get_company_info",
        func=get_company_info,
        description="Get company info like name, sector, description. Args: ticker_symbol (str)."
    ),
    StructuredTool.from_function(
        name="calculate_basic_risk_metrics",
        func=calculate_basic_risk_metrics,
        description="Calculate volatility, Sharpe Ratio and Max Drawdown. Args: ticker_symbol (str)."
    ),
]

system_prompt = """You are an expert financial analyst AI assistant.
Use the available tools to fetch real data when answering questions about stocks.
Always provide structured, concise, and professional analysis.
If charts or tables are already displayed in the app, mention that they are shown above.
Include trend and risk assessment whenever available.
Never make up stock prices or financial data — always use the tools.
"""

class LangGraphAdapter:
    """
    Wraps the LangGraph agent in a simple interface
    that app.py can call with agent.invoke({"input": "..."})

    Why a wrapper class?
    LangGraph returns a dict with a "messages" list.
    This adapter extracts just the final text response
    so app.py doesn't need to know about LangGraph internals.
    """
    def __init__(self, graph):
        self.graph = graph

    def invoke(self, inputs):
        user_input = inputs.get("input", "")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]

        response = self.graph.invoke(
            {"messages": messages},
            {"recursion_limit": 10}
        )

        final_message = response["messages"][-1]
        return {"output": final_message.content}


@st.cache_resource
def create_agent_executor():
    """
    Creates the LangGraph agent with Groq as the LLM.

    Why @st.cache_resource?
    This function creates the agent object which connects
    to the Groq API. We don't want to recreate it on every
    Streamlit rerun — cache_resource keeps it alive for the
    entire session, like a singleton.

    Why Groq instead of Ollama?
    Ollama runs locally — it can't be used when the app is
    deployed to the cloud. Groq is a free cloud API that runs
    the same llama3 models but much faster (typically <1 second
    response time vs 10-30 seconds locally).

    API key handling:
    We read the key from st.secrets first (for cloud deployment)
    and fall back to checking the environment variable (for local).
    This way the same code works both locally and on the cloud
    without any changes.
    """
    # Try Streamlit secrets first (used when deployed on cloud)
    # Fall back to environment variable (used locally)
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        import os
        api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. "
            "Add it to .streamlit/secrets.toml for local use, "
            "or to your cloud platform's environment variables."
        )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # current recommended model for tool use
        temperature=0,             # 0 = deterministic, no hallucination
        api_key=api_key,
    )

    graph = create_react_agent(llm, tools=tools)
    return LangGraphAdapter(graph)