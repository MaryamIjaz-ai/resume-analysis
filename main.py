from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from multiagent_graph import multi_agent_graph

import json
import os
from datetime import datetime

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Agentic AI Resume Assistant",
    description="Resume Analysis + Feedback Monitoring System",
    version="1.0"
)

LOG_FILE = "feedback_log.json"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"


class FeedbackRequest(BaseModel):
    thread_id: str
    feedback: str   # good / bad


# ============================================================
# LOGGING HELPERS
# ============================================================

def load_logs():

    if not os.path.exists(LOG_FILE):
        return []

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_logs(data):

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def log_interaction(user_input, agent_response, thread_id):

    data = load_logs()

    entry = {
        "timestamp": str(datetime.now()),
        "thread_id": thread_id,
        "user_input": user_input,
        "agent_response": agent_response,
        "feedback": None
    }

    data.append(entry)

    save_logs(data)


def update_feedback(thread_id, feedback):

    data = load_logs()

    # Update latest interaction of this thread
    for entry in reversed(data):

        if entry["thread_id"] == thread_id:
            entry["feedback"] = feedback
            break

    save_logs(data)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
async def root():

    return {
        "message": "Agentic AI Resume Assistant Running",
        "docs": "http://127.0.0.1:8000/docs"
    }


# ============================================================
# CHAT ENDPOINT (REAL AI USING OLLAMA + LANGGRAPH)
# ============================================================

@app.post("/chat")
async def chat(req: ChatRequest):

    # ========================================================
    # IMPROVED PROMPT
    # ========================================================
    # FIXES THE ISSUE:
    # Earlier system generated EMAILS instead of resume analysis.
    #
    # Now explicitly restricts the agent behavior.
    # ========================================================

    prompt = f"""
You are an AI Resume Analysis Assistant.

IMPORTANT RULES:
- Analyze the resume ONLY
- DO NOT generate emails
- DO NOT generate cover letters
- DO NOT act as a job application writer
- Return resume analysis only
- Be concise and professional

Your tasks:
1. Extract technical skills
2. Compare with job requirements if provided
3. Calculate ATS compatibility score
4. Identify missing skills
5. Suggest resume improvements

USER INPUT:
{req.message}
"""

    # ========================================================
    # INITIAL STATE
    # ========================================================

    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "current_agent": "none",
        "next_agent": "researcher",
        "sender_email": "",
        "sender_password": "",
        "human_review_done": False
    }

    config = {
        "configurable": {
            "thread_id": req.thread_id
        }
    }

    final_answer = "No response generated."

    # ========================================================
    # RUN MULTI-AGENT GRAPH
    # ========================================================

    try:

        for event in multi_agent_graph.stream(initial_state, config):

            for node, value in event.items():

                if isinstance(value, dict) and "messages" in value:

                    msg = value["messages"][-1]

                    if hasattr(msg, "content") and msg.content:

                        cleaned = msg.content.strip()

                        # Ignore empty responses
                        if cleaned:
                            final_answer = cleaned

    except Exception as e:

        final_answer = f"Error: {str(e)}"

    # ========================================================
    # CLEAN BAD OUTPUTS
    # ========================================================

    # Prevent accidental email responses
    if "Dear Hiring Manager" in final_answer:
        final_answer = (
            "The agent incorrectly attempted to generate an email. "
            "Resume analysis mode is enforced now."
        )

    # ========================================================
    # AUTO LOGGING
    # ========================================================

    log_interaction(
        user_input=req.message,
        agent_response=final_answer,
        thread_id=req.thread_id
    )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "answer": final_answer,
        "status": "completed",
        "thread_id": req.thread_id,
        "feedback_options": [
            "good",
            "bad"
        ]
    }


# ============================================================
# FEEDBACK ENDPOINT
# ============================================================

@app.post("/feedback")
async def feedback(req: FeedbackRequest):

    feedback_value = req.feedback.lower()

    if feedback_value not in ["good", "bad"]:

        return {
            "error": "Feedback must be either 'good' or 'bad'"
        }

    update_feedback(
        thread_id=req.thread_id,
        feedback=feedback_value
    )

    return {
        "message": "Feedback recorded successfully",
        "thread_id": req.thread_id,
        "feedback": feedback_value
    }


# ============================================================
# VIEW LOGS ENDPOINT (OPTIONAL FOR DEBUGGING)
# ============================================================

@app.get("/logs")
async def view_logs():

    data = load_logs()

    return {
        "total_logs": len(data),
        "logs": data
    }