"""
Planner chain — generates a grounded, cited, structured answer.

Route logic
-----------
The planner inspects the query text to decide which prompt template to use:
  • keywords like "plan", "schedule", "semester", "term", "next" → PLAN prompt
  • keywords like "can I take", "eligible", "prereq", "before", "need" → PREREQ prompt
  • everything else → GENERAL prompt

All three prompts enforce the same citation rules and output format.
The output is parsed into a PlannerOutput Pydantic model.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import Citation, PlannerOutput, StudentProfile
from src.prompts.planner_prompt import GENERAL_PROMPT, PLAN_PROMPT, PREREQ_PROMPT
from src.prompts.system_prompt import ABSTENTION_PHRASE

logger = logging.getLogger(__name__)

_PREREQ_RE = re.compile(
    r"\b(can i take|eligible|prereq|prerequisite|before|require|need to|"
    r"ready for|qualify|enroll in|register for)\b",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"\b(plan|schedule|semester|term|next term|fall|spring|suggest|recommend|"
    r"what should i take|courses for)\b",
    re.IGNORECASE,
)


def _pick_prompt(query: str):
    # Check prereq intent first — a direct "can I take X?" question is always
    # a prereq check even when the student also mentions a future term.
    if _PREREQ_RE.search(query):
        return PREREQ_PROMPT
    if _PLAN_RE.search(query):
        return PLAN_PROMPT
    return GENERAL_PROMPT


_CITE_RE = re.compile(
    r"\[Source:\s*([^,\]]+),\s*([^\],]+?)(?:,\s*Section:\s*([^\]]+))?\]",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(Not eligible|Eligible|Need more info)\b", re.IGNORECASE
)


def _safe_parse(raw_text: str) -> Optional[PlannerOutput]:
    """
    Parse the LLM's markdown-sectioned response into PlannerOutput.

    The LLM naturally outputs:
      ## Decision:  ...
      ## Answer / Plan:  ...
      ## Why (requirements/prereqs satisfied):  ...
      ## Citations:  ...
      ## Clarifying questions (if needed):  ...
      ## Assumptions / Not in catalog:  ...

    We split on '##' headings and extract each section, then scrape
    [Source: ...] citation tags from the full text.
    """
    if not raw_text or not raw_text.strip():
        return None

    # ── Split into named sections ──────────────────────────────────────────
    sections: dict[str, str] = {}
    current = "_preamble"
    buf: list[str] = []

    for line in raw_text.split("\n"):
        heading = re.match(r"^##\s+(.+)", line)
        if heading:
            sections[current] = "\n".join(buf).strip()
            current = heading.group(1).strip().lower()
            buf = []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()

    def _get(*keys: str) -> str:
        for k in keys:
            for sec_key, val in sections.items():
                if k.lower() in sec_key:
                    return val
        return ""

    answer        = _get("answer", "plan")
    why           = _get("why", "justif", "prereq", "requirement")
    assumptions   = _get("assumption", "not in catalog")
    clarify_text  = _get("clarify")
    decision_text = _get("decision")

    # If the LLM put everything in one block without headings, use the full text
    if not answer:
        answer = raw_text.strip()

    # ── Extract decision ────────────────────────────────────────────────────
    decision: Optional[str] = None
    search_text = (decision_text + " " + answer).lower()
    if "not eligible" in search_text:
        decision = "Not eligible"
    elif "eligible" in search_text:
        decision = "Eligible"
    elif "need more info" in search_text or "need more information" in search_text:
        decision = "Need more info"

    # ── Extract citations from [Source: title, url, Section: heading] tags ─
    citations: list[Citation] = []
    seen_cites: set[str] = set()
    for m in _CITE_RE.finditer(raw_text):
        title   = m.group(1).strip()
        url     = m.group(2).strip()
        section = (m.group(3) or "General").strip()
        key = f"{title}|{url}"
        if key not in seen_cites:
            seen_cites.add(key)
            citations.append(Citation(title=title, url=url, section=section))

    # ── Extract clarifying questions (lines containing '?') ────────────────
    clarifying: Optional[list[str]] = None
    if clarify_text:
        qs = [
            ln.lstrip("0123456789.-) ").strip()
            for ln in clarify_text.split("\n")
            if "?" in ln and ln.strip()
        ]
        if qs:
            clarifying = qs

    return PlannerOutput(
        decision=decision,
        answer_or_plan=answer,
        why_justified=why,
        citations=citations,
        clarifying_questions=clarifying,
        assumptions_not_in_catalog=assumptions,
    )


def create_planner_chain(llm: BaseLanguageModel):
    """
    Return a callable that produces a ``PlannerOutput`` from context + profile + query.
    """
    def plan(
        query: str,
        context: str,
        profile: StudentProfile,
    ) -> PlannerOutput:
        prompt_template = _pick_prompt(query)
        chain = prompt_template | llm | StrOutputParser()

        try:
            raw_text = chain.invoke(
                {
                    "query":           query,
                    "context":         context,
                    "student_profile": profile.model_dump_json(indent=2),
                }
            )
            result = _safe_parse(raw_text)
            if result is not None:
                return result
        except Exception as exc:
            logger.warning("Planner chain invocation failed: %s", exc)

        # Fallback: safe abstention response
        logger.warning("Returning fallback abstention response.")
        return PlannerOutput(
            decision="Need more info",
            answer_or_plan=ABSTENTION_PHRASE,
            why_justified="The planner encountered an error processing the catalog excerpts.",
            citations=[],
            assumptions_not_in_catalog=(
                "Response generation failed. Please try rephrasing your question "
                "or contact your academic advisor."
            ),
        )

    return plan
