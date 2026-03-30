"""
Shared Pydantic v2 models used across all chains.
Centralised here so every chain imports from one place.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Student profile ───────────────────────────────────────────────────────────

class StudentProfile(BaseModel):
    """Normalised student record collected by the Intake chain."""

    completed_courses: List[str] = Field(
        default_factory=list,
        description="Course codes the student has already passed (e.g. ['COMP1010', 'MATH1310']).",
    )
    grades: Dict[str, str] = Field(
        default_factory=dict,
        description="Optional grade per course, e.g. {'COMP1010': 'B+'}.",
    )
    target_program: Optional[str] = Field(
        default=None,
        description="Degree/program the student is enrolled in, e.g. 'BS Computer Science'.",
    )
    target_term: Optional[str] = Field(
        default=None,
        description="Term the plan is for, e.g. 'Fall 2026'.",
    )
    max_credits: int = Field(
        default=15,
        description="Maximum credit hours the student can take next term.",
    )
    catalog_year: str = Field(
        default="2025-2026",
        description="Catalog year being used for requirements.",
    )


# ── Intake chain output ───────────────────────────────────────────────────────

class IntakeOutput(BaseModel):
    """Result of the Intake chain — either a complete profile or clarifying questions."""

    is_complete: bool = Field(
        description="True when all required fields are present and the profile is ready."
    )
    student_profile: Optional[StudentProfile] = Field(
        default=None,
        description="Populated when is_complete=True.",
    )
    clarifying_questions: Optional[List[str]] = Field(
        default=None,
        description="1–5 questions to ask when is_complete=False.",
    )


# ── Planner chain output ──────────────────────────────────────────────────────

class Citation(BaseModel):
    """A single catalog citation supporting a claim."""

    title: str = Field(description="Document title, e.g. 'COMP 4110 – Database Systems'.")
    url: str = Field(description="Full URL of the source page.")
    section: str = Field(description="Section heading within the page, e.g. 'Prerequisites'.")

    def format(self) -> str:
        return f"[Source: {self.title}, {self.url}, Section: {self.section}]"


class PlannerOutput(BaseModel):
    """Structured response produced by the Planner chain."""

    decision: Optional[str] = Field(
        default=None,
        description="'Eligible' | 'Not eligible' | 'Need more info' — for prereq questions.",
    )
    answer_or_plan: str = Field(
        description="The main answer text or proposed course plan."
    )
    why_justified: str = Field(
        description="Reasoning: how requirements/prereqs are satisfied or not."
    )
    citations: List[Citation] = Field(
        default_factory=list,
        description="All catalog citations backing the claims above.",
    )
    clarifying_questions: Optional[List[str]] = Field(
        default=None,
        description="Follow-up questions if more info is needed before a decision.",
    )
    assumptions_not_in_catalog: str = Field(
        default="",
        description="Items assumed or explicitly not found in the catalog documents.",
    )

    def to_markdown(self) -> str:
        """Render the output in the mandatory assessment format."""
        lines: List[str] = []

        if self.decision:
            lines.append(f"**Decision:** {self.decision}\n")

        lines.append("## Answer / Plan:")
        lines.append(str(self.answer_or_plan or ""))

        lines.append("\n## Why (requirements/prereqs satisfied):")
        lines.append(str(self.why_justified or ""))

        lines.append("\n## Citations:")
        if self.citations:
            for c in self.citations:
                lines.append(f"- {c.format()}")
        else:
            lines.append("- *(none — see Assumptions section)*")

        if self.clarifying_questions:
            lines.append("\n## Clarifying questions (if needed):")
            for i, q in enumerate(self.clarifying_questions, 1):
                lines.append(f"{i}. {q}")

        lines.append("\n## Assumptions / Not in catalog:")
        lines.append(str(
            self.assumptions_not_in_catalog
            or "None — all claims are grounded in the retrieved catalog excerpts."
        ))

        return "\n".join(lines)


# ── Verifier chain output ─────────────────────────────────────────────────────

class VerifierOutput(BaseModel):
    """Audit result from the Verifier chain."""

    verified: bool = Field(
        description="True when the planner output passes all citation and logic checks."
    )
    issues: List[str] = Field(
        default_factory=list,
        description="List of flagged problems; empty when verified=True.",
    )
    corrected_answer: Optional[str] = Field(
        default=None,
        description="Corrected answer_or_plan text when verified=False.",
    )
