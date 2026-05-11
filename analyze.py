import json
from collections import Counter

LOG_FILE = "feedback_log.json"

with open(LOG_FILE, "r") as f:
    data = json.load(f)

total = len(data)
negative = sum(1 for d in data if d["feedback"] == "bad")

print("Total Responses:", total)
print("Negative Feedback:", negative)

# Top failed queries
failed_queries = [d["user_input"] for d in data if d["feedback"] == "bad"]

top_3 = Counter(failed_queries).most_common(3)

print("\nTop 3 Failed Queries:")
for query, count in top_3:
    print(f"{query} → {count} times")