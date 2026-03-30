"""
app.py — Gradio web demo for the Catalog RAG assistant.

Usage
-----
    python app.py
    # Opens at http://localhost:7860

Layout
------
  Left sidebar  : Student Profile form (courses, grades, program, term, credits)
  Right main    : Chat interface with formatted responses and citation badges
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Pipeline — loaded once at server startup, shared across all requests ──────
_pipeline = None


def _init_pipeline():
    """Load embeddings + FAISS + LLM once at startup so the first chat request
    is fast (~3-5s) instead of bearing the full 60-90s model-init cost."""
    global _pipeline
    from langchain_groq import ChatGroq

    from src.chains.pipeline import Pipeline
    from src.embeddings.embedder import get_embeddings
    from src.vector_store.faiss_store import index_exists, load_vectorstore

    if not index_exists():
        raise RuntimeError("FAISS index not found. Run: python build_index.py")

    print("Loading embeddings model …", flush=True)
    embeddings  = get_embeddings()
    print("Loading FAISS index …", flush=True)
    vectorstore = load_vectorstore(embeddings)
    llm = ChatGroq(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
        api_key=config.GROQ_API_KEY,
    )
    _pipeline = Pipeline(vectorstore, llm)
    print("Pipeline ready — all requests will now be fast.", flush=True)


def _get_pipeline():
    return _pipeline


# ── Chat logic ────────────────────────────────────────────────────────────────

def _build_full_query(
    user_message: str,
    completed_courses: str,
    grades: str,
    target_program: str,
    target_term: str,
    max_credits: int,
) -> str:
    """Combine sidebar profile with the chat message into one prompt."""
    parts: List[str] = []

    if target_program:
        parts.append(f"I am enrolled in {target_program}.")
    if completed_courses.strip():
        parts.append(f"Completed courses: {completed_courses.strip()}.")
    if grades.strip():
        parts.append(f"Grades: {grades.strip()}.")
    if target_term:
        parts.append(f"Planning for {target_term}.")
    parts.append(f"Max credits: {max_credits}.")
    parts.append(f"\n{user_message}")

    return " ".join(parts)


def chat(
    user_message: str,
    history: list,
    completed_courses: str,
    grades: str,
    target_program: str,
    target_term: str,
    max_credits: int,
) -> tuple:
    """Main chat callback — uses Gradio 6.x messages format."""
    if not user_message.strip():
        return history, ""

    full_query = _build_full_query(
        user_message, completed_courses, grades, target_program, target_term, max_credits
    )

    try:
        pipeline = _get_pipeline()
        result   = pipeline.run(full_query)
        response = result.to_markdown()
    except RuntimeError as exc:
        response = f"⚠️ **Setup error:** {exc}"
    except Exception as exc:
        logger.exception("Pipeline error")
        response = (
            f"⚠️ **Error:** {exc}\n\n"
            "Please try rephrasing your question, or contact your academic advisor."
        )

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": response},
    ]
    return history, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

import gradio as gr  # noqa: E402 (import after sys.path setup)

_CSS = """
#chatbot .message.bot { background: #f0f4ff; border-left: 4px solid #4a6cf7; }
#chatbot .message.user { background: #f9fafb; }
.citation-box { font-size: 0.85em; color: #555; margin-top: 8px; }
footer { display: none !important; }
"""

_TITLE = "📚 UML Course Planning Assistant"
_DESCRIPTION = (
    "Answers prerequisite questions and builds semester plans "
    "**strictly grounded in the UML academic catalog**. "
    "Every claim is cited. Unavailable information is acknowledged honestly."
)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title=_TITLE) as demo:

        gr.Markdown(f"# {_TITLE}\n{_DESCRIPTION}")

        with gr.Row(equal_height=False):

            # ── Left sidebar: Student Profile ────────────────────────────────
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### 👤 Student Profile")
                gr.Markdown(
                    "*Fill in what you know. Leave fields blank to be asked.*"
                )

                target_program = gr.Dropdown(
                    choices=[
                        "BS Computer Science",
                        "CS Minor",
                        "MS Computer Science",
                        "Other",
                    ],
                    label="Program / Degree",
                    value="BS Computer Science",
                )
                target_term = gr.Dropdown(
                    choices=[
                        "Fall 2026", "Spring 2027",
                        "Fall 2027", "Spring 2028",
                    ],
                    label="Target Term",
                    value="Fall 2026",
                )
                max_credits = gr.Slider(
                    minimum=3, maximum=21, step=3,
                    value=15, label="Max Credits",
                )
                completed_courses = gr.Textbox(
                    label="Completed Courses",
                    placeholder="One per line or comma-separated",
                    lines=3,
                )
                grades = gr.Textbox(
                    label="Grades (optional)",
                    placeholder="Optional — course:grade pairs",
                    lines=2,
                )

            # ── Right main: Chat ──────────────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Conversation",
                    height=520,
                    render_markdown=True,
                )

                with gr.Row():
                    msg_box = gr.Textbox(
                        label="Your question",
                        placeholder="Ask about prerequisites, planning, or program requirements…",
                        scale=4,
                        container=False,
                    )
                    send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑 Clear Chat", size="sm")

                gr.Markdown(
                    "<small>⚠️ This assistant is grounded in the UML catalog only. "
                    "Course availability and live schedules are not in scope — "
                    "always confirm with your advisor.</small>"
                )

        # ── Wire events ───────────────────────────────────────────────────────
        submit_inputs  = [msg_box, chatbot, completed_courses, grades,
                          target_program, target_term, max_credits]
        submit_outputs = [chatbot, msg_box]

        msg_box.submit(fn=chat, inputs=submit_inputs, outputs=submit_outputs)
        send_btn.click(fn=chat, inputs=submit_inputs, outputs=submit_outputs)
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_box])

    return demo


if __name__ == "__main__":
    if not config.GROQ_API_KEY:
        print(
            "\n[ERROR] GROQ_API_KEY not set.\n"
            "  Copy .env.example → .env and add your key.\n"
        )
        sys.exit(1)

    # Load the pipeline at startup — first chat message will then be fast
    _init_pipeline()

    ui = build_ui()
    print(
        f"Starting Gradio on port {config.GRADIO_SERVER_PORT} … "
        f"(if bind fails, set GRADIO_SERVER_PORT=7861 in .env or stop the old process)",
        flush=True,
    )
    ui.launch(
        server_name="0.0.0.0",
        server_port=config.GRADIO_SERVER_PORT,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(primary_hue="indigo"),
    )
