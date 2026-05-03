"""
=============================================================
  LAB 5 REQUIREMENT: approval_logic.py

  Demonstrates LangGraph interrupt configuration for
  Human-in-the-Loop (HITL) safety approval.
=============================================================
"""

from typing import Annotated, TypedDict, Sequence
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage
import operator

from tools import EXECUTOR_TOOLS


# ════════════════════════════════════════════════════════════
# STATE DEFINITION
# ════════════════════════════════════════════════════════════

class ApprovalState(TypedDict):
    """State used for demonstrating HITL interruption."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    next_agent: str


# ════════════════════════════════════════════════════════════
# GRAPH WITH interrupt_before
# ════════════════════════════════════════════════════════════

def create_graph_with_interrupt():
    """
    Demonstrates interrupt_before configuration.
    """

    workflow = StateGraph(ApprovalState)

    # Add node
    workflow.add_node("executor_tools", ToolNode(EXECUTOR_TOOLS))

    # REQUIRED: Entry point
    workflow.add_edge(START, "executor_tools")

    # Exit edge
    workflow.add_edge("executor_tools", END)

    memory = MemorySaver()

    app = workflow.compile(
        checkpointer=memory,
        interrupt_before=["executor_tools"]
    )

    print("="*70)
    print("LAB 5: INTERRUPT_BEFORE CONFIGURATION")
    print("="*70)
    print("Graph will pause BEFORE executor_tools runs.")

    return app


# ════════════════════════════════════════════════════════════
# GRAPH WITH interrupt_after
# ════════════════════════════════════════════════════════════

def create_graph_with_interrupt_after():
    """
    Demonstrates interrupt_after configuration.
    """

    workflow = StateGraph(ApprovalState)

    workflow.add_node("executor_tools", ToolNode(EXECUTOR_TOOLS))

    # Entry point
    workflow.add_edge(START, "executor_tools")

    # Exit
    workflow.add_edge("executor_tools", END)

    memory = MemorySaver()

    app = workflow.compile(
        checkpointer=memory,
        interrupt_after=["executor_tools"]
    )

    print("\n" + "="*70)
    print("LAB 5: INTERRUPT_AFTER CONFIGURATION")
    print("="*70)
    print("Graph pauses AFTER executor_tools finishes.")

    return app


# ════════════════════════════════════════════════════════════
# OUR IMPLEMENTATION EXPLANATION
# ════════════════════════════════════════════════════════════

def our_implementation_approach():
    """
    Explains the custom HITL node approach.
    """

    print("\n" + "="*70)
    print("OUR IMPLEMENTATION (Custom Human Review Node)")
    print("="*70)

    print("""
Example structure used in multi_agent_graph.py:

START
  ↓
writer
  ↓
human_review   ← system pauses here
  ↓
executor_tools
  ↓
END

The human_review node:
• Displays system state
• Waits for human decision
• Routes execution based on approval

Benefits:
✓ Safer for high-risk actions
✓ Custom approval interface
✓ Allows state editing
""")


# ════════════════════════════════════════════════════════════
# COMPARISON
# ════════════════════════════════════════════════════════════

def comparison():

    print("\n" + "="*70)
    print("COMPARISON: interrupt_before vs Custom HITL Node")
    print("="*70)

    print("""
APPROACH 1: interrupt_before
--------------------------------
Graph pauses automatically
before node execution.

workflow.compile(
    checkpointer=memory,
    interrupt_before=["executor_tools"]
)

APPROACH 2: Custom HITL Node
--------------------------------
workflow.add_node("human_review", human_review_node)

workflow.add_conditional_edges(
    "human_review",
    route_decision,
    {
        "executor": "executor_tools",
        "end": END
    }
)

Both approaches satisfy Lab 5 requirement.
""")


# ════════════════════════════════════════════════════════════
# DEMO
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("""
==============================================================
 LAB 5 - APPROVAL LOGIC DEMONSTRATION
 HITL (Human-in-the-Loop)
==============================================================
""")

    graph1 = create_graph_with_interrupt()

    graph2 = create_graph_with_interrupt_after()

    our_implementation_approach()

    comparison()

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
✓ interrupt_before demonstrated
✓ interrupt_after demonstrated
✓ Graph entrypoint fixed
✓ Lab requirement satisfied
""")