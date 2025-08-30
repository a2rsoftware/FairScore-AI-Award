# graph_basic.py
from typing import Annotated, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# ---- State ----
class GraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    plan: Optional[str]

# ---- LLM (Ollama local if available) ----
try:
    from langchain_ollama import ChatOllama
    llm = ChatOllama(model="phi3:mini", base_url="http://localhost:11434")
except Exception:
    class Dummy:
        def invoke(self, msgs):
            return AIMessage(content="Hello from Dummy LLM (Ollama not running).")
    llm = Dummy()

# ---- Helpers ----
def _content_of(m) -> str:
    if hasattr(m, "content"):
        return m.content or ""
    if isinstance(m, dict):
        return m.get("content", "") or ""
    return str(m or "")

def _is_human(m) -> bool:
    if isinstance(m, HumanMessage):
        return True
    if isinstance(m, dict):
        return m.get("role") in ("human", "user")
    return getattr(m, "type", None) == "human"

# ---- Nodes ----
def plan_node(state: GraphState):
    user_text = ""
    for m in reversed(state["messages"]):
        if _is_human(m):
            user_text = _content_of(m)
            break
    prompt = f"Create a 3-step plan to answer the user.\n\nUser: {user_text}"
    reply = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=_content_of(reply))], "plan": "3-step plan generated"}

def answer_node(state: GraphState):
    full_context = "\n".join(_content_of(m) for m in state["messages"] if _content_of(m))
    prompt = f"Using the conversation so far, answer clearly in 5 bullets:\n\n{full_context}"
    reply = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=_content_of(reply))]}

# ---- Build graph (builder stays module-level) ----
builder = StateGraph(GraphState)
builder.add_node("planner", plan_node)
builder.add_node("answer", answer_node)
builder.set_entry_point("planner")
builder.add_edge("planner", "answer")
builder.add_edge("answer", END)

def build_graph(checkpointer=None):
    """Return a compiled graph. Pass a checkpointer for persistence."""
    return builder.compile(checkpointer=checkpointer)

# Optional: default compiled graph (no persistence) for simple imports
graph = build_graph()

if __name__ == "__main__":
    # Optional local test
    from langgraph.checkpoint.sqlite import SqliteSaver
    cfg = {"configurable": {"thread_id": "demo-thread-1"}}
    init = {"messages": [HumanMessage(content="Summarize the benefits of LangGraph on Windows.")]}

    try:
        with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
            g = build_graph(checkpointer=cp)
            print("Streaming events...\n")
            for event in g.stream(init, config=cfg):
                for node, payload in event.items():
                    msgs = payload.get("messages", [])
                    if msgs:
                        print(f"[{node}] -> {_content_of(msgs[-1])}")
            print("\nDone with SQLite persistence.")
    except Exception as e:
        print(f"Running without persistence ({e})")
        g = build_graph()
        for event in g.stream(init, config=cfg):
            for node, payload in event.items():
                msgs = payload.get("messages", [])
                if msgs:
                    print(f"[{node}] -> {_content_of(msgs[-1])}")
        print("\nDone.")