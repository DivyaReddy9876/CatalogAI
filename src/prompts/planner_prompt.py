"""
Planner chain prompt templates.
"""
from langchain_core.prompts import ChatPromptTemplate

from src.prompts.system_prompt import (
    GROUNDING_RULES,
    CITATION_REMINDER,
    COMPACT_FORMAT_INSTRUCTIONS,
)

# ── Prerequisite / eligibility question prompt ────────────────────────────────
PREREQ_SYSTEM = (
    GROUNDING_RULES
    + "\n"
    + CITATION_REMINDER
    + """
TASK: Prerequisite / Eligibility Check
---------------------------------------
The student is asking whether they can enrol in a course.

Steps you MUST follow:
1. Look up the course's prerequisites in the catalog excerpts.
2. Compare each prerequisite against the student's completed courses and grades.
3. Check minimum grade requirements stated in the catalog.
4. Produce the structured output below.

If ANY required prerequisite is missing from the student's record, the
decision is "Not eligible" — even if all other prereqs are satisfied.

"""
    + COMPACT_FORMAT_INSTRUCTIONS
)

PREREQ_HUMAN = """\
CATALOG EXCERPTS:
{context}

STUDENT PROFILE:
{student_profile}

STUDENT QUESTION:
{query}
"""

PREREQ_PROMPT = ChatPromptTemplate.from_messages(
    [("system", PREREQ_SYSTEM), ("human", PREREQ_HUMAN)]
)


# ── Course plan generation prompt ─────────────────────────────────────────────
PLAN_SYSTEM = (
    GROUNDING_RULES
    + "\n"
    + CITATION_REMINDER
    + """
TASK: Semester Course Plan
--------------------------
The student wants a course plan for their target term.

Steps you MUST follow:
1. Identify which BS CS core requirements and electives remain based on their
   completed courses and the program requirements in the excerpts.
2. For each candidate course, verify ALL prerequisites are satisfied.
3. Respect the student's max_credits limit.
4. Propose 3–5 courses that: (a) fit requirement categories, (b) have prereqs
   met, and (c) form a coherent next step toward graduation.
5. For each proposed course state:
   - Why it fits requirements (cite program requirement page)
   - How its prerequisites are satisfied (cite course page)
6. In the Assumptions section, note anything NOT in the catalog
   (e.g. actual course availability, section availability).

"""
    + COMPACT_FORMAT_INSTRUCTIONS
)

PLAN_HUMAN = """\
CATALOG EXCERPTS:
{context}

STUDENT PROFILE:
{student_profile}

STUDENT QUESTION:
{query}
"""

PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [("system", PLAN_SYSTEM), ("human", PLAN_HUMAN)]
)


# ── General / policy question prompt ─────────────────────────────────────────
GENERAL_SYSTEM = (
    GROUNDING_RULES
    + "\n"
    + CITATION_REMINDER
    + """
TASK: General Catalog Question
-------------------------------
Answer the student's question using ONLY the provided catalog excerpts.
Apply all mandatory rules above, especially Rule 3 (safe abstention).

"""
    + COMPACT_FORMAT_INSTRUCTIONS
)

GENERAL_HUMAN = """\
CATALOG EXCERPTS:
{context}

STUDENT PROFILE:
{student_profile}

STUDENT QUESTION:
{query}
"""

GENERAL_PROMPT = ChatPromptTemplate.from_messages(
    [("system", GENERAL_SYSTEM), ("human", GENERAL_HUMAN)]
)
