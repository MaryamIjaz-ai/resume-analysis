from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(model="llama3.2", temperature=0)


def run_eval_llm(query: str) -> str:

    prompt = f"""
You are an AI assistant.

Answer clearly and directly.

RULES:
- Be specific
- No generic answers
- Keep it short (1-2 sentences)

Question:
{query}
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return response.content.strip()