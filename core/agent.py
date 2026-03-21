import streamlit as st
from langchain_ollama import ChatOllama
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

tools = [
    StructuredTool.from_function(
        name="get_stock_price_chart",
        func=get_stock_price_chart,
        description="Get historical stock price chart. Args: ticker_symbol (str), period (str)."
    ),
    StructuredTool.from_function(
        name="get_fundamental_data",
        func=get_fundamental_data,
        description="Get financial fundamentals. Args: ticker_symbol (str)."
    ),
    StructuredTool.from_function(
        name="get_price_forecast",
        func=get_price_forecast,
        description="Get AI price forecast. Args: ticker_symbol (str)."
    ),
    StructuredTool.from_function(
        name="get_company_info",
        func=get_company_info,
        description="Get company info. Args: ticker_symbol (str)."
    ),
    StructuredTool.from_function(
        name="calculate_basic_risk_metrics",
        func=calculate_basic_risk_metrics,
        description="Calculate volatility, trend and risk level. Args: ticker_symbol (str)."
    ),
]

system_prompt = """You are an expert financial analyst AI.
Use tools when necessary.
Provide structured, concise and professional analysis.
If charts or tables are generated, they are already displayed.
Include trend and risk assessment when available.
"""

class LangGraphAdapter:
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

        return {
            "output": final_message.content
        }

@st.cache_resource
def create_agent_executor():
    llm = ChatOllama(
        model="llama3.2",
        temperature=0
    )

    graph = create_react_agent(llm, tools=tools)

    return LangGraphAdapter(graph)
