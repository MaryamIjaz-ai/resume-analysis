"""
=============================================================
  Lab 8 - FastAPI + Streaming (FINAL VERSION)
=============================================================
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import asyncio
import json

from schema import ChatRequest, ChatResponse
from multiagent_graph import multi_agent_graph
from langchain_core.messages import HumanMessage


# ════════════════════════════════════════════════════════════
# LIFESPAN (Global Graph - Persistence)
# ════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[FastAPI] Starting up...")

    # Store graph globally (persistent memory)
    app.state.graph = multi_agent_graph

    yield

    print("[FastAPI] Shutting down...")


app = FastAPI(lifespan=lifespan)


# ════════════════════════════════════════════════════════════
# TASK 2 — NORMAL CHAT ENDPOINT
# ════════════════════════════════════════════════════════════

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    graph = app.state.graph

    initial_state = {
        "messages": [HumanMessage(content=request.message)],
        "current_agent": "none",
        "next_agent": "researcher",
        "sender_email": "",
        "sender_password": ""
    }

    config = {
        "configurable": {
            "thread_id": request.thread_id
        }
    }

    final_output = ""

    for event in graph.stream(initial_state, config):
        for node, value in event.items():
            if "messages" in value:
                msg = value["messages"][-1]

                if hasattr(msg, "content") and msg.content:
                    final_output = msg.content

    return ChatResponse(
        final_answer=final_output,
        status="completed"
    )


# ════════════════════════════════════════════════════════════
# TASK 3 — STREAMING GENERATOR
# ════════════════════════════════════════════════════════════

async def stream_generator(message: str, thread_id: str):
    """Streams LangGraph output using Server-Sent Events (SSE)."""

    graph = app.state.graph

    initial_state = {
        "messages": [HumanMessage(content=message)],
        "current_agent": "none",
        "next_agent": "researcher",
        "sender_email": "",
        "sender_password": ""
    }

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    async for event in graph.astream(initial_state, config):
        for node, value in event.items():

            if "messages" in value:
                msg = value["messages"][-1]

                if hasattr(msg, "content") and msg.content:
                    content = msg.content.strip()

                    if content:
                        data = {
                            "node": node,
                            "content": content
                        }

                        yield f"data: {json.dumps(data)}\n\n"

        await asyncio.sleep(0.01)  # smooth streaming

    yield "data: [DONE]\n\n"


# ════════════════════════════════════════════════════════════
# TASK 3 — STREAM ENDPOINT
# ════════════════════════════════════════════════════════════

@app.post("/stream")
async def stream_chat(request: ChatRequest):

    return StreamingResponse(
        stream_generator(request.message, request.thread_id),
        media_type="text/event-stream"
    )