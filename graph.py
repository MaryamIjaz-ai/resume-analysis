"""
=============================================================
  Lab 3 – The Reasoning Loop (LangGraph)
  File: graph.py
  Project: Agentic AI Resume & Job Application Assistant
=============================================================

TASK 2: Defining the Graph State & Nodes
TASK 3: The Conditional Router

This implements a ReAct (Reason + Act) loop using LangGraph where:
- The agent THINKS about what tool to use
- ACTS by calling tools
- REASONS about the results
- REPEATS until it has enough info to answer

NO API KEY REQUIRED - Uses local Ollama LLM.
=============================================================
"""

import operator
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import ChatOllama

# Import our custom tools from tools.py
from tools import ALL_TOOLS


# ════════════════════════════════════════════════════════════
#  TASK 2: DEFINE THE GRAPH STATE
# ════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    The state of our ReAct agent.
    
    This TypedDict stores the conversation history and allows the agent
    to maintain context across multiple reasoning steps.
    
    Attributes:
        messages: List of all messages in the conversation.
                  Uses Annotated to specify that new messages should be
                  added to the list (not replace it).
    """
    messages: Annotated[list[BaseMessage], operator.add]


# ════════════════════════════════════════════════════════════
#  LLM SETUP - Using Local Ollama (FREE)
# ════════════════════════════════════════════════════════════

print("[Graph] Initializing ChatOllama LLM...")
print("        Make sure Ollama is running: ollama serve")
print("        Required model: llama3.2 (run: ollama pull llama3.2)")

# Initialize Ollama chat model (local, no API key)
llm = ChatOllama(
    model="llama3.2",           # Fast, free local model
    temperature=0,              # Deterministic for consistency
    base_url="http://localhost:11434"  # Ollama default port
)

# Bind tools to the LLM - this teaches the LLM what tools are available
llm_with_tools = llm.bind_tools(ALL_TOOLS)

print("[Graph] LLM initialized with tools.\n")


# ════════════════════════════════════════════════════════════
#  TASK 2: DEFINE THE AGENT NODE
# ════════════════════════════════════════════════════════════

def agent_node(state: AgentState) -> AgentState:
    """
    The Agent Node - where the LLM reasons and decides what to do next.
    
    This function:
    1. Takes the current state (conversation history)
    2. Passes it to the LLM
    3. The LLM decides to either:
       - Call a tool (returns tool_calls)
       - Give a final answer (returns text response)
    4. Returns updated state with LLM's message
    
    Args:
        state: Current conversation state
    
    Returns:
        Updated state with LLM's response added
    """
    print("\n[Agent Node] LLM is thinking...")
    
    # Get all messages so far
    messages = state["messages"]
    
    # Call LLM with full conversation history
    response = llm_with_tools.invoke(messages)
    
    # Debug: Show what LLM decided
    if response.tool_calls:
        print(f"[Agent Node] LLM decided to call {len(response.tool_calls)} tool(s)")
        for tc in response.tool_calls:
            print(f"             - {tc['name']}")
    else:
        print("[Agent Node] LLM decided to give final answer")
    
    # Return updated state with LLM's message appended
    return {"messages": [response]}


# ════════════════════════════════════════════════════════════
#  TASK 2: DEFINE THE TOOL NODE
# ════════════════════════════════════════════════════════════

# LangGraph provides ToolNode out-of-the-box - it automatically:
# 1. Extracts tool calls from the LLM's message
# 2. Executes those tools with the provided arguments
# 3. Returns tool results back into the state
tool_node = ToolNode(ALL_TOOLS)


# ════════════════════════════════════════════════════════════
#  TASK 3: THE CONDITIONAL ROUTER
# ════════════════════════════════════════════════════════════

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    The Router - decides what happens next based on LLM's response.
    
    This is the "control flow logic gate" that makes the ReAct loop work.
    
    Logic:
    - If LLM generated tool_calls → route to "tools" (execute tools, loop back)
    - If LLM generated text response → route to "end" (stop, we have final answer)
    
    Args:
        state: Current state with messages
    
    Returns:
        "tools" if we should execute tools, "end" if we're done
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("[Router] → Routing to TOOLS node")
        return "tools"
    else:
        print("[Router] → Routing to END (final answer ready)")
        return "end"


# ════════════════════════════════════════════════════════════
#  BUILD THE GRAPH
# ════════════════════════════════════════════════════════════

def create_agent_graph():
    """
    Build the LangGraph StateGraph for our ReAct agent.
    
    Graph structure:
    
        START
          ↓
       [Agent]  ← Thinks & decides
          ↓
       [Router] ← Conditional edge
        ↙   ↘
    [Tools]  [END]
       ↓
    [Agent] (loops back)
    
    Returns:
        Compiled LangGraph graph ready to invoke
    """
    print("[Graph] Building StateGraph...")
    
    # Initialize graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edge from agent
    # The router decides: "tools" or "end"
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # If tools needed → go to tool node
            "end": END         # If done → stop
        }
    )
    
    # Add edge from tools back to agent (creates the loop)
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    print("[Graph] Graph compiled successfully!")
    print("[Graph] Structure: START → agent → router → [tools ⟲ agent] → END\n")
    
    return app


# ════════════════════════════════════════════════════════════
#  CREATE THE GRAPH INSTANCE
# ════════════════════════════════════════════════════════════

agent_graph = create_agent_graph()


# ════════════════════════════════════════════════════════════
#  HELPER FUNCTION: Run the Agent
# ════════════════════════════════════════════════════════════

def run_agent(user_query: str, verbose: bool = True) -> str:
    """
    Run the agent with a user query.
    
    This is a convenience wrapper that:
    1. Formats the user query as a HumanMessage
    2. Invokes the graph
    3. Streams the reasoning process (if verbose=True)
    4. Returns the final answer
    
    Args:
        user_query: The user's question or request
        verbose: Whether to print reasoning steps
    
    Returns:
        Final answer from the agent
    """
    print("=" * 60)
    print("  Running Agent")
    print("=" * 60)
    print(f"User Query: {user_query}\n")
    
    # Create initial state with user message
    initial_state = {
        "messages": [HumanMessage(content=user_query)]
    }
    
    # Invoke the graph (this runs the ReAct loop)
    # stream() returns intermediate states, invoke() returns final state
    if verbose:
        for event in agent_graph.stream(initial_state):
            for node_name, output in event.items():
                if "messages" in output:
                    last_msg = output["messages"][-1]
                    print(f"\n[{node_name.upper()}]")
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print("  Tool calls:")
                        for tc in last_msg.tool_calls:
                            print(f"    - {tc['name']}({tc.get('args', {})})")
                    elif isinstance(last_msg, ToolMessage):
                        print(f"  Tool result: {last_msg.content[:200]}...")
                    else:
                        print(f"  Response: {last_msg.content[:300]}...")
    else:
        result = agent_graph.invoke(initial_state)
    
    # Get final result
    final_state = agent_graph.invoke(initial_state)
    final_message = final_state["messages"][-1]
    
    print("\n" + "=" * 60)
    print("  FINAL ANSWER")
    print("=" * 60)
    print(final_message.content)
    print("=" * 60 + "\n")
    
    return final_message.content


# ════════════════════════════════════════════════════════════
#  MAIN - Test the Graph
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    =============================================================
      Lab 3 - LangGraph ReAct Agent
      Agentic AI Resume & Job Application Assistant
    =============================================================
    
    This agent can:
    - Query the knowledge base (from Lab 2)
    - Extract skills from resumes
    - Calculate skill match scores
    - Calculate ATS compatibility scores
    - Generate improvement suggestions
    
    The agent will autonomously decide which tools to use based on
    your query.
    
    =============================================================
    """)
    
    # Example test queries
    test_queries = [
        "What are the ATS formatting rules I should follow for my resume?",
        
        "I have Python, JavaScript, and React skills on my resume. "
        "The job requires Python, React, Docker, and Kubernetes. "
        "Calculate my skill match score.",
    ]
    
    print("Choose a test query:")
    for i, q in enumerate(test_queries, 1):
        print(f"  {i}. {q[:80]}...")
    print(f"  {len(test_queries)+1}. Custom query")
    
    try:
        choice = int(input("\nEnter choice (1-3): "))
        if 1 <= choice <= len(test_queries):
            query = test_queries[choice - 1]
        else:
            query = input("Enter your custom query: ")
        
        run_agent(query, verbose=True)
    
    except KeyboardInterrupt:
        print("\n\nAgent execution cancelled.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("  1. Ollama is running: ollama serve")
        print("  2. Model is pulled: ollama pull llama3.2")
        print("  3. Lab 2 ChromaDB exists: ./chroma_db")