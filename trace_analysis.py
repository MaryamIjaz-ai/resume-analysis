"""
=============================================================
  Lab 7 - Task 3: Trace-Based Bottleneck Analysis
  - LangSmith Tracing Enabled
  - Runs 5 complex queries
  - Prints latency + failure insights
=============================================================
"""

import time
from datetime import datetime
from langchain_core.messages import HumanMessage

# Import your EXISTING graph
from multiagent_graph import multi_agent_graph
from dotenv import load_dotenv
load_dotenv()


# ════════════════════════════════════════════════════════════
# TEST QUERIES (COMPLEX)
# ════════════════════════════════════════════════════════════

queries = [
    "Analyze my resume and calculate ATS score for a Data Scientist job",
    "Extract skills and calculate match against job requiring Python, SQL, AWS",
    "Suggest improvements and generate a professional cover email",
    "Evaluate resume, suggest fixes, and prepare job application email",
    "Full pipeline: analyze, match, improve, and generate email"
]


# ════════════════════════════════════════════════════════════
# RUN WITH TRACING
# ════════════════════════════════════════════════════════════

def run_with_tracing(query: str, thread_id: str):
    print("\n" + "=" * 70)
    print(f"Running Query: {query}")
    print("=" * 70)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_agent": "none",
        "next_agent": "researcher",
        "sender_email": "test@example.com",     # dummy
        "sender_password": "dummy"              # dummy
    }

    config = {
        "configurable": {"thread_id": thread_id},
        "run_name": f"trace_{thread_id}"   # 👈 appears in LangSmith
    }

    start_time = time.time()

    try:
        for _ in multi_agent_graph.stream(initial_state, config):
            pass

        status = "SUCCESS"

    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        status = "FAILED"

    end_time = time.time()
    latency = end_time - start_time

    print(f"\n⏱️ Total Latency: {latency:.2f} seconds")
    print(f"Status: {status}")

    return {
        "query": query,
        "latency": latency,
        "status": status
    }


# ════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    =============================================================
      Lab 7 - Task 3: Trace-Based Analysis
      LangSmith Tracing ENABLED
    =============================================================
    """)

    results = []

    for i, query in enumerate(queries, 1):
        thread_id = f"trace_{datetime.now().strftime('%H%M%S')}_{i}"
        result = run_with_tracing(query, thread_id)
        results.append(result)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for r in results:
        print(f"\nQuery: {r['query']}")
        print(f"Latency: {r['latency']:.2f}s | Status: {r['status']}")

    print("\n\n👉 Now open LangSmith dashboard to analyze traces:")
    print("https://smith.langchain.com")