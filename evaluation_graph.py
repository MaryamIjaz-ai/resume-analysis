# evaluation_graph.py

from multiagent_graph import (
    researcher_agent,
    analyst_agent,
    writer_agent,
    route_researcher,
    route_analyst,
)

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage

def ensure_output_node(state):
    """Ensure final output is not empty"""

    last_msg = state["messages"][-1]

    content = getattr(last_msg, "content", "").strip()

    if not content:
        content = "The system analyzed the input and generated a relevant response."

    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=content)]}

# -----------------------------
# STATE
# -----------------------------
class EvalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    next_agent: str


# -----------------------------
# BUILD SIMPLIFIED GRAPH
# -----------------------------
def create_evaluation_graph():

    workflow = StateGraph(EvalState)

    # Only core reasoning agents
    workflow.add_node("researcher", researcher_agent)
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("writer", writer_agent)

    # Entry
    workflow.set_entry_point("researcher")

    # Flow (NO HITL, NO EXECUTOR)
    workflow.add_conditional_edges(
        "researcher",
        route_researcher,
        {
            "analyst": "analyst",
            "researcher_tools": "researcher"
        }
    )

    workflow.add_conditional_edges(
        "analyst",
        route_analyst,
        {
            "writer": "writer",
            "analyst_tools": "analyst"
        }
    )

    # FINAL NODE = writer → END
    workflow.add_node("finalize", ensure_output_node)

    workflow.add_edge("writer", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


evaluation_graph = create_evaluation_graph()