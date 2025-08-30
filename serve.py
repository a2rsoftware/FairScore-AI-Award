# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite import SqliteSaver
from graph_basic import build_graph  # we'll compile at startup

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Open the SQLite checkpointer for the life of the app
    with SqliteSaver.from_conn_string("checkpoints.sqlite") as cp:
        app.state.graph = build_graph(checkpointer=cp)
        yield
    # context exits -> checkpointer closed

app = FastAPI(lifespan=lifespan)

class RunInput(BaseModel):
    state: Dict[str, Any]          # e.g. {"messages":[{"role":"user","content":"hi"}]}
    thread_id: str = "thread-1"

@app.post("/run")
def run_graph(inp: RunInput):
    g = app.state.graph
    config = {"configurable": {"thread_id": inp.thread_id}}
    # drive the graph to completion (non-streaming)
    for _ in g.stream(inp.state, config=config):
        pass
    final = g.get_state(config).values
    return {"state": final}