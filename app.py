import streamlit as st
import requests
import sqlite3
import uuid
from datetime import datetime

# =========================================
# DATABASE
# =========================================

conn = sqlite3.connect("feedback_log.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    thread_id TEXT,
    message_id TEXT,
    user_input TEXT,
    agent_response TEXT,
    feedback_score INTEGER,
    optional_comment TEXT
)
""")

conn.commit()

# =========================================
# STREAMLIT PAGE
# =========================================

st.set_page_config(page_title="Agentic Resume Assistant")

st.title("Agentic AI Resume Assistant")

# =========================================
# SESSION STATE
# =========================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================
# USER INPUT
# =========================================

user_prompt = st.chat_input("Enter your resume or job description")

# =========================================
# PROCESS INPUT
# =========================================

if user_prompt:

    with st.spinner("Thinking..."):

        try:
            response = requests.post(
                "http://127.0.0.1:8000/chat",
                json={
                    "message": user_prompt,
                    "thread_id": st.session_state.thread_id
                }
            )

            data = response.json()

            # IMPORTANT
            assistant_reply = data.get("answer", "No response received.")

        except Exception as e:
            assistant_reply = f"Error: {str(e)}"

    # Save in session
    st.session_state.chat_history.append({
        "message_id": str(uuid.uuid4()),
        "user": user_prompt,
        "assistant": assistant_reply
    })

# =========================================
# DISPLAY CHAT HISTORY
# =========================================

for i, chat in enumerate(st.session_state.chat_history):

    # USER MESSAGE
    with st.chat_message("user"):
        st.markdown(chat["user"])

    # ASSISTANT MESSAGE
    with st.chat_message("assistant"):

        # THIS SHOWS RESPONSE ON FRONTEND
        st.markdown(chat["assistant"])

        # =================================
        # FEEDBACK
        # =================================

        col1, col2 = st.columns(2)

        feedback_score = None

        with col1:
            if st.button("👍 Good", key=f"good_{i}"):
                feedback_score = 1

        with col2:
            if st.button("👎 Bad", key=f"bad_{i}"):
                feedback_score = -1

        comment = st.text_input(
            "Optional Comment",
            key=f"comment_{i}"
        )

        # =================================
        # SAVE FEEDBACK
        # =================================

        if feedback_score is not None:

            cursor.execute("""
            INSERT INTO feedback_logs (
                timestamp,
                thread_id,
                message_id,
                user_input,
                agent_response,
                feedback_score,
                optional_comment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(datetime.now()),
                st.session_state.thread_id,
                chat["message_id"],
                chat["user"],
                chat["assistant"],
                feedback_score,
                comment
            ))

            conn.commit()

            st.success("Feedback Saved!")