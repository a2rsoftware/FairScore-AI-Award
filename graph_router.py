# graph_router.py
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ---- State ----
class S(TypedDict):
    # LangGraph will store BaseMessage objects here
    messages: Annotated[List[BaseMessage], add_messages]

# ---- Helpers ----
def _content_of(m) -> str:
    """Return message text for BaseMessage objects or dicts."""
    if hasattr(m, "content"):
        return m.content or ""
    if isinstance(m, dict):
        return m.get("content", "") or ""
    return str(m or "")

# ---- Router (classifier) ----
def classify_node(state: S) -> str:
    last_msg = state["messages"][-1]
    text = _content_of(last_msg).lower()
    return "planner" if "plan" in text else "answer"

# ---- Nodes ----
def planner(state: S) -> S:
    return {
        "messages": [
            AIMessage(
                content="High-level plan:\n1) Gather info\n2) Draft\n3) Validate"
            )
        ]
    }

def answer(state: S) -> S:
    return {"messages": [AIMessage(content="Here is your concise answer.")]}

# ---- Build graph ----
g = StateGraph(S)

# IMPORTANT: add the router node before wiring conditional edges
def router(state: S) -> S:
    return state

g.add_node("router", router)
g.add_node("planner", planner)
g.add_node("answer", answer)

g.set_entry_point("router")
g.add_conditional_edges("router", classify_node, {"planner": "planner", "answer": "answer"})
g.add_edge("planner", END)
g.add_edge("answer", END)

graph = g.compile()

if __name__ == "__main__":
    # You can pass dicts or message objects; objects shown here:
    initial = {"messages": [HumanMessage(content="Give me a plan.")]}

    for event in graph.stream(initial):
        for node, payload in event.items():
            msgs = payload.get("messages", [])
            if msgs:
                print(f"[{node}] -> {_content_of(msgs[-1])}")