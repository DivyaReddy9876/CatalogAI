"""
Post-process LLM markdown for Gradio / common Markdown renderers.

Some models wrap URLs in double underscores (__url__), which Markdown treats as
emphasis and breaks clickable links. We strip those and normalize bare URLs.
"""
from __future__ import annotations

import re

# __https://example.com/path__  →  https://example.com/path
_RE_UNDERSCORE_WRAPPED_URL = re.compile(
    r"__(https?://[^\s\)<>\]]+?)__",
    re.IGNORECASE,
)

# Optional: (https://...) wrapped only on the outside
_RE_PAREN_UNDERSCORE_URL = re.compile(
    r"__\((https?://[^\s\)]+)\)__",
    re.IGNORECASE,
)


def sanitize_markdown_output(text: str) -> str:
    """Remove mistaken __ wrappers around URLs; keep plain https URLs clickable."""
    if not text:
        return text
    s = _RE_UNDERSCORE_WRAPPED_URL.sub(r"\1", text)
    s = _RE_PAREN_UNDERSCORE_URL.sub(r"(\1)", s)
    return s
