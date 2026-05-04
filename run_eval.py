"""
=============================================================
  CI-READY EVALUATION SCRIPT (FINAL)
  - Relevancy + Tool Accuracy
  - Reads thresholds from JSON
  - Writes results.json
  - Exits with code 0 / 1
=============================================================
"""

import json
import sys
import os

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

from ollama_deepeval_model import OllamaDeepEval
from evaluation_llm import run_eval_llm


# ════════════════════════════════════════════════════════════
# LOAD DATASET
# ════════════════════════════════════════════════════════════

with open("test_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

dataset = dataset[:3]

with open("eval_thresholds.json") as f:
    thresholds = json.load(f)

REL_THRESHOLD = thresholds["relevancy"]
TOOL_THRESHOLD = thresholds["tool_accuracy"]

# ════════════════════════════════════════════════════════════
# LOAD THRESHOLDS
# ════════════════════════════════════════════════════════════

with open("eval_thresholds.json", "r") as f:
    thresholds = json.load(f)

REL_THRESHOLD = thresholds["relevancy"]
TOOL_THRESHOLD = thresholds["tool_accuracy"]


# ════════════════════════════════════════════════════════════
# RUN AGENT
# ════════════════════════════════════════════════════════════

def run_agent(query: str) -> str:
    return "This is completely incorrect output"



# ════════════════════════════════════════════════════════════
# TOOL ACCURACY
# ════════════════════════════════════════════════════════════

def expected_tool(query: str) -> str:
    q = query.lower()

    if "extract skills" in q:
        return "extract_skills_from_resume"
    elif "analyze resume" in q:
        return "extract_skills_from_resume"
    elif "match" in q:
        return "calculate_skill_match_score"
    elif "ats" in q:
        return "calculate_ats_score"
    elif "improve" in q:
        return "generate_improvement_suggestions"
    elif "email" in q:
        return "send_resume_to_employer"
    else:
        return "none"


def predicted_tool(output: str) -> str:
    o = output.lower()

    if "skill" in o:
        return "extract_skills_from_resume"
    elif "match" in o:
        return "calculate_skill_match_score"
    elif "ats" in o:
        return "calculate_ats_score"
    elif "improve" in o:
        return "generate_improvement_suggestions"
    elif "email" in o or "dear" in o:
        return "send_resume_to_employer"
    else:
        return "none"


# ════════════════════════════════════════════════════════════
# INIT MODEL
# ════════════════════════════════════════════════════════════

print("[Eval] Initializing model...")
def simple_relevancy(actual, expected):
    actual = actual.lower()
    expected = expected.lower()

    match_count = 0
    for word in expected.split():
        if word in actual:
            match_count += 1

    return match_count / len(expected.split())


# ════════════════════════════════════════════════════════════
# RUN TESTS
# ════════════════════════════════════════════════════════════

scores = []
tool_correct = 0
tool_total = 0

for item in dataset:
    query = item["query"]
    expected = item["ground_truth"]

    try:
        output = run_agent(query)
    except Exception as e:
        output = f"ERROR: {str(e)}"

    # Relevancy
    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        expected_output=expected
    )

    try:
        score = simple_relevancy(output, expected)
        scores.append(score)
        print(f"Relevancy Score: {score:.2f}")
    except:
        scores.append(0)

    # Tool Accuracy
    exp_tool = expected_tool(query)
    pred_tool = predicted_tool(output)

    if exp_tool == pred_tool:
        tool_correct += 1

    tool_total += 1


# ════════════════════════════════════════════════════════════
# FINAL SCORES
# ════════════════════════════════════════════════════════════

avg_relevancy = sum(scores) / len(scores) if scores else 0
tool_accuracy = tool_correct / tool_total if tool_total else 0

print(f"Relevancy: {avg_relevancy:.2f}")
print(f"Tool Accuracy: {tool_accuracy:.2f}")


# ════════════════════════════════════════════════════════════
# SAVE RESULTS (CI REQUIRED)
# ════════════════════════════════════════════════════════════

results = {
    "metrics": {
        "relevancy": {
            "score": avg_relevancy,
            "threshold": REL_THRESHOLD,
            "pass": avg_relevancy >= REL_THRESHOLD
        },
        "tool_accuracy": {
            "score": tool_accuracy,
            "threshold": TOOL_THRESHOLD,
            "pass": tool_accuracy >= TOOL_THRESHOLD
        }
    },
    "overall_pass": avg_relevancy >= REL_THRESHOLD and tool_accuracy >= TOOL_THRESHOLD
}

with open("results.json", "w") as f:
    json.dump(results, f, indent=4)

# Exit code for CI
if results["overall_pass"]:
    print("\n✅ QUALITY GATE PASSED")
    sys.exit(0)
else:
    print("\n❌ QUALITY GATE FAILED")
    sys.exit(1)