import os
from functools import lru_cache

from dotenv import load_dotenv

# Load environment variables (like GROQ_API_KEY or OPENAI_API_KEY)
load_dotenv()


def _get_secret(key: str):
    """Read a secret from env vars, .env file, or Streamlit Cloud secrets."""
    value = os.getenv(key)
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_llm():
    """
    Initializes and returns the selected LLM instance.
    Prefers Groq when GROQ_API_KEY is available, otherwise falls back to OpenAI.
    """
    groq_api_key = _get_secret("GROQ_API_KEY")
    openai_api_key = _get_secret("OPENAI_API_KEY")

    if groq_api_key:
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=groq_api_key,
            model=_get_secret("GROQ_MODEL") or "llama-3.3-70b-versatile",
            temperature=float(_get_secret("LLM_TEMPERATURE") or "0.2"),
        )

    if openai_api_key:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=openai_api_key,
            model=_get_secret("OPENAI_MODEL") or "gpt-4o-mini",
            temperature=float(_get_secret("LLM_TEMPERATURE") or "0.2"),
        )

    raise RuntimeError(
        "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in your environment or .env file."
    )


def test_llm_connection():
    """
    Simple function to test if the LLM connection is functioning properly.
    """
    print("Testing LLM Connection...")
    try:
        llm = get_llm()
        response = llm.invoke("Hello, this is a test from the Agentic AI Fleet Management system. Reply with 'Connection Successful!' if you receive this.")
        print(f"LLM Reply: {response.content}")
        print("Test Passed.")
    except Exception as e:
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_llm_connection()