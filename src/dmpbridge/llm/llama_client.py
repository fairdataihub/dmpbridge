# src/dmpbridge/llm/llama_client.py

from langchain_ollama import ChatOllama


def load_llama(model_name="llama3.1:8b", temperature=0):
    return ChatOllama(
        model=model_name,
        temperature=temperature,
    )