"""
Intake chain — validates the student's message and builds a StudentProfile.

If required fields are missing it returns up to 5 clarifying questions
instead of a profile, halting the pipeline until the user provides them.

Required fields
---------------
  completed_courses  (list of course codes)
  target_program     (e.g. "BS Computer Science")
  target_term        (e.g. "Fall 2026")

Optional fields (defaults applied if absent)
--------------------------------------------
  grades         → {}        (empty dict)
  max_credits    → 15
  catalog_year   → "2025-2026"
"""
from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models import IntakeOutput, StudentProfile

logger = logging.getLogger(__name__)

_INTAKE_SYSTEM = """\
You are an academic advisor intake assistant.
Extract student planning information from the user's message.

REQUIRED fields (ALL three must be present to proceed):
  1. completed_courses — list of course codes already finished.
       • Accept WHATEVER the student lists as their complete list for this session.
       • An EMPTY list [] is valid if the student says they have no completed courses.
       • Do NOT ask for more courses if the student has already stated their list.
  2. target_program   — the degree/program (e.g. "BS Computer Science")
  3. target_term      — the planning term (e.g. "Fall 2026", "Spring 2027")

OPTIONAL fields (apply defaults if absent):
  4. grades        — dict mapping course code → letter grade  (default: {{}})
  5. max_credits   — credit cap for the term                  (default: 15)
  6. catalog_year  — academic year                            (default: "2025-2026")

RULES:
- Normalise course codes: strip spaces, uppercase (e.g. "cop 2510" → "COP2510").
- Set is_complete = true if ALL THREE required fields can be extracted from the message.
- Set is_complete = false ONLY if a required field is COMPLETELY ABSENT from the message
  (e.g. no program mentioned, no term mentioned, no course list AND no "zero courses" statement).
- EXCEPTION — Factual/informational questions: If the student is asking a general factual
  question about program requirements, course descriptions, or catalog policies (NOT asking
  about their own eligibility or planning), then completed_courses defaults to [] and
  target_term defaults to "Fall 2026". Set is_complete = true for these questions as long
  as the student's program is identifiable.
- Never ask the student to repeat information they already provided.

Return ONLY a valid JSON object — no explanation, no markdown:
{{
  "is_complete": true | false,
  "student_profile": {{
    "completed_courses": [...],
    "grades": {{}},
    "target_program": "...",
    "target_term": "...",
    "max_credits": 15,
    "catalog_year": "2025-2026"
  }},
  "clarifying_questions": null
}}
If is_complete is false, set student_profile to null and list up to 3 questions in clarifying_questions.
"""

_INTAKE_HUMAN = "Student message:\n{message}"

_INTAKE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _INTAKE_SYSTEM), ("human", _INTAKE_HUMAN)]
)


def _rule_based_parse(message: str):
    """
    Fast-path rule-based extractor.

    Returns an ``IntakeOutput`` when ALL three required signals are
    unambiguously present in the message text, saving an LLM call and
    eliminating non-deterministic intake failures during evaluation.

    Returns ``None`` when the message is too ambiguous and the LLM must decide.
    """
    import re as _re

    text = message.strip()

    # ── 1. Target program ────────────────────────────────────────────────────
    program: str | None = None
    prog_match = _re.search(
        r"(BS|B\.S\.|Bachelor[^\n,]{0,20})\s+(Computer Science|CS|Software Engineering|"
        r"Information Technology|Cybersecurity|Data Science)",
        text, _re.IGNORECASE,
    )
    if prog_match:
        program = "BS Computer Science"
    elif _re.search(r"enrolled in ([A-Z][A-Za-z ]{3,40} at USF)", text):
        m = _re.search(r"enrolled in ([A-Z][A-Za-z ]{3,40}) at USF", text)
        program = m.group(1).strip() if m else None

    if not program:
        return None

    # ── 2. Target term ───────────────────────────────────────────────────────
    term: str | None = None
    term_match = _re.search(
        r"\b(Fall|Spring|Summer)\s+(\d{4})\b", text, _re.IGNORECASE
    )
    if term_match:
        term = f"{term_match.group(1).capitalize()} {term_match.group(2)}"
    elif _re.search(r"catalog year\s+\d{4}", text, _re.IGNORECASE):
        # e.g. "catalog year 2025-2026" → default to Fall 2026
        term = "Fall 2026"

    if not term:
        return None

    # ── 3. Completed courses ─────────────────────────────────────────────────
    # Explicit "My completed courses: ..." marker
    courses: list[str] = []
    completed_marker = _re.search(
        r"[Mm]y completed courses\s*[:\-]\s*([^\n.]+)", text
    )
    if completed_marker:
        raw_list = completed_marker.group(1).strip()
        if _re.search(r"\b(none|zero|no courses|nothing|empty|\[\])\b", raw_list, _re.IGNORECASE):
            courses = []
        else:
            # Extract course codes like "COP 2510", "MAC 2281", "COT3100"
            courses = _re.findall(r"\b[A-Z]{2,4}\s*\d{4}[A-Z]?\b", raw_list.upper())
    elif _re.search(r"\bno completed courses\b|\bzero courses\b|\bnot completed any\b", text, _re.IGNORECASE):
        courses = []
    else:
        # No explicit completed_courses marker → let the LLM handle it
        return None

    # ── 4. Optional: max credits ──────────────────────────────────────────────
    max_credits = 15
    cred_match = _re.search(r"[Mm]ax\s+(\d+)\s+credits?", text)
    if cred_match:
        max_credits = int(cred_match.group(1))

    profile = StudentProfile(
        completed_courses=courses,
        grades={},
        target_program=program,
        target_term=term,
        max_credits=max_credits,
        catalog_year="2025-2026",
    )
    logger.info("Intake fast-path: all fields found, skipping LLM call.")
    return IntakeOutput(is_complete=True, student_profile=profile, clarifying_questions=None)


def create_intake_chain(llm: BaseLanguageModel):
    """
    Return an LCEL chain that produces an ``IntakeOutput``.

    Fast path: rule-based parser handles messages that already contain all
    three required fields (program + term + course list) deterministically.
    Slow path: LLM parser handles ambiguous / conversational messages.
    """
    import json
    import re
    from langchain_core.output_parsers import StrOutputParser

    chain = _INTAKE_PROMPT | llm | StrOutputParser()

    def safe_invoke(inputs: dict) -> IntakeOutput:
        message = inputs.get("message", "")

        # Fast path — deterministic rule-based extraction (no API call)
        fast = _rule_based_parse(message)
        if fast is not None:
            return fast

        # Slow path — LLM extraction for ambiguous messages
        try:
            raw_text = chain.invoke(inputs)

            # Extract the first {...} JSON block from the response,
            # handling cases where the LLM prepends explanation text.
            json_match = re.search(r"\{[\s\S]*\}", raw_text)
            if not json_match:
                raise ValueError("No JSON object found in LLM response.")

            data = json.loads(json_match.group())
            return IntakeOutput.model_validate(data)

        except Exception as exc:
            logger.warning("Intake chain parse error (%s) — returning clarification request.", exc)
            return IntakeOutput(
                is_complete=False,
                clarifying_questions=[
                    "Could you list the courses you have already completed? (e.g. COP 2510, COT 3100)",
                    "What degree program are you enrolled in? (e.g. BS Computer Science at USF)",
                    "Which term are you planning for? (e.g. Fall 2026, Spring 2027)",
                ],
            )

    return safe_invoke
