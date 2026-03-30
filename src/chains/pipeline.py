"""
Pipeline orchestrator — wires the four chains into a single callable.

Flow
----
User message
    │
    ▼
[1] Intake chain
    ├── is_complete=False → return clarifying questions (stop)
    └── is_complete=True  → StudentProfile built
            │
            ▼
    [2] Retrieval chain
        → fetch top-k catalog chunks (MultiQueryRetriever + MMR)
            │
            ▼
    [3] Planner chain
        → generate structured PlannerOutput with citations
            │
            ▼
    [4] Verifier chain
        ├── verified=True  → return final PlannerOutput
        └── verified=False → retry Planner once with correction prompt
                │
                ├── retry passes  → return corrected output
                └── retry fails   → return original + disclaimer

The ``run`` function is the only public surface; everything else is internal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseLanguageModel

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.chains.intake_chain     import create_intake_chain
from src.chains.planner_chain    import create_planner_chain
from src.chains.retrieval_chain  import create_retrieval_chain
from src.chains.verifier_chain   import create_verifier_chain
from src.models                  import IntakeOutput, PlannerOutput, StudentProfile
from src.utils.markdown_sanitize import sanitize_markdown_output

logger = logging.getLogger(__name__)

_DISCLAIMER = (
    "\n\n---\n"
    "⚠️  **Note:** One or more claims in this response could not be fully "
    "verified against the catalog excerpts. Please confirm details with "
    "your academic advisor before making enrollment decisions."
)


@dataclass
class PipelineResult:
    """Typed return value from ``Pipeline.run``."""

    response_type: str          # "clarify" | "answer"
    planner_output: Optional[PlannerOutput] = None
    clarifying_questions: Optional[list] = None
    verified: bool = True
    issues: list = None         # populated when verified=False

    def __post_init__(self):
        if self.issues is None:
            self.issues = []

    def to_markdown(self) -> str:
        if self.response_type == "clarify":
            lines = ["I need a few more details before I can help you plan:\n"]
            for i, q in enumerate(self.clarifying_questions or [], 1):
                lines.append(f"{i}. {q}")
            return sanitize_markdown_output("\n".join(lines))

        md = self.planner_output.to_markdown()
        if not self.verified:
            md += _DISCLAIMER
        return sanitize_markdown_output(md)


class Pipeline:
    """
    Encapsulates all four chains and the vectorstore.
    Instantiate once per process; call ``run()`` for every student message.
    """

    def __init__(self, vectorstore: FAISS, llm: BaseLanguageModel) -> None:
        self._intake    = create_intake_chain(llm)
        self._retrieve  = create_retrieval_chain(vectorstore, llm)
        self._plan      = create_planner_chain(llm)
        self._verify, self._correct = create_verifier_chain(llm)
        logger.info("Pipeline initialised (4 chains ready).")

    def run(self, message: str) -> PipelineResult:
        """
        Process one student message end-to-end.

        Parameters
        ----------
        message : Raw student text (may contain profile info or just a question).

        Returns
        -------
        PipelineResult with response_type="clarify" or "answer".
        """
        # ── Step 1: Intake ────────────────────────────────────────────────────
        logger.info("Step 1/4 — Intake chain")
        intake: IntakeOutput = self._intake({"message": message})

        if not intake.is_complete:
            logger.info("Intake incomplete — returning clarifying questions.")
            return PipelineResult(
                response_type="clarify",
                clarifying_questions=intake.clarifying_questions,
            )

        profile: StudentProfile = intake.student_profile
        logger.info(
            "Profile: program=%s, term=%s, completed=%d courses",
            profile.target_program, profile.target_term, len(profile.completed_courses),
        )

        # ── Step 2: Retrieval ─────────────────────────────────────────────────
        logger.info("Step 2/4 — Retrieval chain")
        docs, context = self._retrieve(message, profile)

        if not docs:
            logger.warning("No catalog chunks retrieved — returning abstention.")
            from src.prompts.system_prompt import ABSTENTION_PHRASE
            return PipelineResult(
                response_type="answer",
                planner_output=PlannerOutput(
                    answer_or_plan=ABSTENTION_PHRASE,
                    why_justified="No relevant catalog excerpts were retrieved.",
                    citations=[],
                    assumptions_not_in_catalog=(
                        "Could not find relevant catalog content for this query. "
                        "Please contact your academic advisor."
                    ),
                ),
                verified=True,
            )

        # ── Step 3: Planning ──────────────────────────────────────────────────
        logger.info("Step 3/4 — Planner chain")
        planner_output: PlannerOutput = self._plan(
            query=message, context=context, profile=profile
        )

        # ── Step 4: Verification ──────────────────────────────────────────────
        logger.info("Step 4/4 — Verifier chain")
        verifier_result = self._verify(planner_output, context, profile, message)

        if verifier_result.verified:
            return PipelineResult(
                response_type="answer",
                planner_output=planner_output,
                verified=True,
            )

        # Retry once with corrections
        logger.info("Verifier found %d issue(s) — retrying planner.", len(verifier_result.issues))
        corrected_text = self._correct(
            issues=verifier_result.issues,
            context=context,
            profile=profile,
            query=message,
        )

        if corrected_text:
            planner_output.answer_or_plan = corrected_text
            logger.info("Correction applied successfully.")
            return PipelineResult(
                response_type="answer",
                planner_output=planner_output,
                verified=True,
            )

        # Retry also failed — return original with disclaimer
        logger.warning("Correction failed — returning original answer with disclaimer.")
        return PipelineResult(
            response_type="answer",
            planner_output=planner_output,
            verified=False,
            issues=verifier_result.issues,
        )
