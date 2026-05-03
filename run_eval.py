"""
=============================================================
  Lab 7 - FINAL EVALUATION (WITH TOOL ACCURACY)
  - No async errors
  - Works with Ollama
  - Relevancy + Tool Accuracy
=============================================================
"""

import json
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from ollama_deepeval_model import OllamaDeepEval
from evaluation_llm import run_eval_llm


# ════════════════════════════════════════════════════════════
# LOAD DATASET
# ════════════════════════════════════════════════════════════

with open("test_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Keep small for stability
dataset = dataset[:3]


# ════════════════════════════════════════════════════════════
# RUN AGENT
# ════════════════════════════════════════════════════════════

def run_agent(query: str) -> str:
    return run_eval_llm(query)


# ════════════════════════════════════════════════════════════
# TOOL ACCURACY HELPERS
# ════════════════════════════════════════════════════════════

def expected_tool(query: str) -> str:
    q = query.lower()

    if "extract skills" in q:
        return "extract_skills_from_resume"

    elif "analyze resume" in q or "analyze my resume" in q:
        return "extract_skills_from_resume"

    elif "skill match" in q or "match" in q:
        return "calculate_skill_match_score"

    elif "ats" in q:
        return "calculate_ats_score"

    elif "improve" in q or "suggest" in q:
        return "generate_improvement_suggestions"

    elif "email" in q or "cover" in q:
        return "send_resume_to_employer"

    else:
        return "none"


def predicted_tool(output: str) -> str:
    o = output.lower()

    if "resume" in o and "skill" in o:
        return "extract_skills_from_resume"

    elif "match" in o:
        return "calculate_skill_match_score"

    elif "ats" in o:
        return "calculate_ats_score"

    elif "improve" in o or "suggest" in o:
        return "generate_improvement_suggestions"

    elif "email" in o or "dear hiring manager" in o:
        return "send_resume_to_employer"

    else:
        return "none"


# ════════════════════════════════════════════════════════════
# METRIC INIT
# ════════════════════════════════════════════════════════════

print("[Eval] Initializing model...")

judge_model = OllamaDeepEval()
relevancy_metric = AnswerRelevancyMetric(model=judge_model)

print("[Eval] ✓ Ready\n")


# ════════════════════════════════════════════════════════════
# RUN TESTS
# ════════════════════════════════════════════════════════════

test_cases = []

tool_correct = 0
tool_total = 0

print("=" * 70)
print("RUNNING TEST CASES")
print("=" * 70)

for i, item in enumerate(dataset, 1):
    query = item["query"]
    expected = item["ground_truth"]

    print(f"\n🔹 Test Case {i}")
    print(f"Query: {query}")

    try:
        output = run_agent(query)
    except Exception as e:
        output = f"ERROR: {str(e)}"

    print("\n--- EXPECTED OUTPUT ---")
    print(expected)

    print("\n--- ACTUAL OUTPUT ---")
    print(output)

    # TOOL ACCURACY
    exp_tool = expected_tool(query)
    pred_tool = predicted_tool(output)

    print(f"\nExpected Tool: {exp_tool}")
    print(f"Predicted Tool: {pred_tool}")

    if exp_tool == pred_tool:
        tool_correct += 1

    tool_total += 1

    print("-" * 70)

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output=expected
    )

    test_cases.append(test_case)


# ════════════════════════════════════════════════════════════
# CALCULATE RELEVANCY
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("CALCULATING METRICS")
print("=" * 70)

scores = []

for i, test_case in enumerate(test_cases, 1):
    print(f"\n🔹 Evaluating Test Case {i}")

    try:
        score = relevancy_metric.measure(test_case)
        scores.append(score)
        print(f"Relevancy Score: {score:.2f}")
    except Exception as e:
        print(f"Error: {e}")


# ════════════════════════════════════════════════════════════
# FINAL RESULTS
# ════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

# Relevancy
if scores:
    avg_rel = sum(scores) / len(scores)
    print(f"Average Relevancy Score: {avg_rel:.2f}")

# Tool Accuracy
if tool_total > 0:
    tool_acc = tool_correct / tool_total
    print(f"Tool Call Accuracy: {tool_acc:.2f}")

print("\n✅ Evaluation completed successfully")