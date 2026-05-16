

import json
import os
import urllib.parse
import urllib.request
from typing import TypedDict

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langchain.agents import create_agent


load_dotenv()

groq_api_key = os.getenv("GROQ_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

if not groq_api_key or not langsmith_api_key:
    raise ValueError("Set GROQ_API_KEY and LANGSMITH_API_KEY in .env")

os.environ["GROQ_API_KEY"] = groq_api_key
os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "langgraph-hackai"


# -------------------- load LLM --------------------
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# -------------------- define tools --------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a basic expression like 12*(3+4)."""
    allowed = set("0123456789+-*/(). ")
    if any(ch not in allowed for ch in expression):
        return "Invalid expression."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception:
        return "Could not calculate."


@tool
def web_search(query: str) -> str:
    """Simple web search using DuckDuckGo instant answer API."""
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("AbstractText") or data.get("Heading") or "No result found."


# -------------------- create agent --------------------
react_agent = create_agent(llm, tools=[calculator, web_search])


def call_agent(query: str) -> str:
    out = react_agent.invoke({"messages": [HumanMessage(content=query)]})
    return out["messages"][-1].content


# -------------------- run agent --------------------
print("Run agent:")
print(call_agent("What is 15 * 9?"))


# -------------------- add memory --------------------
memory = InMemorySaver()


# -------------------- add branching --------------------
class AppState(TypedDict):
    query: str
    answer: str


def route(state: AppState) -> str:
    q = state["query"].lower()
    if any(ch.isdigit() for ch in q) or "search" in q or "who is" in q:
        return "agent"
    return "skip"


def agent_node(state: AppState) -> AppState:
    return {"query": state["query"], "answer": call_agent(state["query"])}


def skip_node(state: AppState) -> AppState:
    return {"query": state["query"], "answer": "Skipped by branch rule."}


builder = StateGraph(AppState)
builder.add_node("agent", agent_node)
builder.add_node("skip", skip_node)
builder.add_conditional_edges(START, route, {"agent": "agent", "skip": "skip"})
builder.add_edge("agent", END)
builder.add_edge("skip", END)
graph = builder.compile(checkpointer=memory)

cfg = {"configurable": {"thread_id": "demo-thread-1"}}
print("\nBranch + memory run 1:")
print(graph.invoke({"query": "100/4 + 2", "answer": ""}, config=cfg)["answer"])
print("\nBranch + memory run 2 (same thread_id):")
print(graph.invoke({"query": "hello", "answer": ""}, config=cfg)["answer"])
