# ollama_deepeval_model.py

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama


class OllamaDeepEval(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatOllama(model="llama3.2", temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "ollama-llama3.2"