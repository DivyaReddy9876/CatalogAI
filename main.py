"""
main.py — interactive CLI for the Catalog RAG assistant.

Usage
-----
    python main.py

Type your question at the prompt.  Type 'quit' or press Ctrl-C to exit.
Type 'reset' to start a fresh conversation.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

logging.basicConfig(
    level=logging.WARNING,         # quiet during interactive use
    format="%(levelname)s %(message)s",
)

_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         UML COURSE PLANNING ASSISTANT  (CLI)                 ║
║  Powered by LangChain · FAISS · Groq Llama-3.1-8b           ║
╠══════════════════════════════════════════════════════════════╣
║  Type your question and press Enter.                         ║
║  Commands:  'reset' — new session   'quit' — exit           ║
╚══════════════════════════════════════════════════════════════╝
"""

_SEPARATOR = "─" * 66


def _load_pipeline():
    """Load resources and return a ready Pipeline instance."""
    from langchain_groq import ChatGroq

    from src.chains.pipeline import Pipeline
    from src.embeddings.embedder import get_embeddings
    from src.vector_store.faiss_store import index_exists, load_vectorstore

    if not index_exists():
        print(
            "\n[ERROR] FAISS index not found.\n"
            "  Run:  python build_index.py\n"
            "  Then retry:  python main.py\n"
        )
        sys.exit(1)

    if not config.GROQ_API_KEY:
        print(
            "\n[ERROR] GROQ_API_KEY is not set.\n"
            "  Copy .env.example → .env and add your key.\n"
        )
        sys.exit(1)

    print("Loading resources …", end="", flush=True)
    embeddings  = get_embeddings()
    vectorstore = load_vectorstore(embeddings)
    llm         = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key=config.GROQ_API_KEY,
    )
    pipeline = Pipeline(vectorstore, llm)
    print(" done.\n")
    return pipeline


def main() -> None:
    print(_BANNER)

    pipeline = _load_pipeline()
    conversation: list[str] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if user_input.lower() == "reset":
            conversation.clear()
            print("[Session reset.]\n")
            continue

        # Build full message including prior turns for context
        conversation.append(user_input)
        full_message = "\n".join(conversation[-4:])   # last 4 turns = 2 exchanges

        result = pipeline.run(full_message)
        response = result.to_markdown()

        print(f"\nAssistant:\n{_SEPARATOR}")
        print(response)
        print(f"{_SEPARATOR}\n")

        # Add assistant response to conversation history
        conversation.append(f"[Assistant]: {response[:300]}")


if __name__ == "__main__":
    main()
