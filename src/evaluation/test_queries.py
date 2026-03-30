"""
25-query evaluation set — University of South Florida (USF) catalog.

Categories
----------
  Cat 1 — Prerequisite checks     (queries 1–10,  10 queries)
  Cat 2 — Prerequisite chains     (queries 11–15,  5 queries)
  Cat 3 — Program requirements    (queries 16–20,  5 queries)
  Cat 4 — Not-in-docs / trick     (queries 21–25,  5 queries)

USF course codes used (all confirmed in data/processed/):
  COP 2510  Programming Concepts          (entry-level; prereq: math placement)
  COP 2513  Object Oriented Programming   (prereq: COP 2510 min C)
  COP 3514  Program Design                (prereq: COP 2510 min B AND MAC 2281 min B)
  COP 3515  Advanced Program Design       (prereq: COP 2513 min B AND 4 math/sci min B)
  COP 4538  Data Structures & Algorithms  (prereq: COP 3515 min C)
  COP 4703  Advanced Database Systems     (prereq: COP 4538 min C)
  COP 4710  Database Design               (prereq: COP 4530 — not in our docs)
  COP 4600  Operating Systems             (prereq: COP 4530 — not in our docs)
  COP 4020  Programming Languages         (prereq: COP 4530 — not in our docs)
  CEN 4020  Software Engineering          (prereq: COP 4530 min C — capstone attribute)
  COT 3100  Intro to Discrete Structures  (prereq: MAC 2281)
  COT 4400  Analysis of Algorithms        (prereq: COP 4530 — not in our docs)
  Computer_Science_B_S_C_S               (120 total degree hours)
  Academic_Policies_and_Procedures

OCR note: the raw extracted text sometimes reads "CQP" for "COP" due to OCR error
on the O/Q character. Queries below use the correct "COP" codes; retrieval is
still functional because the embeddings capture semantic similarity.

Each entry is a dict with:
  id               : 1-based integer
  category         : 1 | 2 | 3 | 4
  query            : the student's question (verbatim input to pipeline)
  expected_decision: "Eligible" | "Not eligible" | "Need more info" | None
  expected_abstain : True if the correct response is an abstention
  grading_notes    : rubric guidance for manual review
"""
from __future__ import annotations

from typing import Any, Dict, List

TestQuery = Dict[str, Any]

# ── Shared student context strings ────────────────────────────────────────────

_CTX_FRESH = (
    "I am enrolled in BS Computer Science at USF. "
    "I have completed NO courses yet (brand new student). "
    "I am planning for Fall 2026. Max 15 credits."
)

_CTX_MID = (
    "I am enrolled in BS Computer Science at USF. "
    "Completed: COP 2510 (grade B+), COP 2513 (grade A), "
    "MAC 2281 (grade B+), COT 3100 (grade A). "
    "Planning for Fall 2026. Max 15 credits."
)

_CTX_SENIOR = (
    "I am enrolled in BS Computer Science at USF. "
    "Completed: COP 2510, COP 2513, COP 3514, COP 3515, COP 4538, "
    "COT 3100, COT 4400, MAC 2281. "
    "Planning for Fall 2026. Max 15 credits."
)

_CTX_MATH = (
    "I am enrolled in BS Computer Science at USF. "
    "Completed: MAC 2281 (grade A). "
    "Planning for Spring 2027. Max 12 credits."
)

# ── The 25 test queries ───────────────────────────────────────────────────────

TEST_QUERIES: List[TestQuery] = [

    # ── Category 1: Prerequisite checks (10) ──────────────────────────────────

    {
        "id": 1,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "I have completed COP 2510 with a grade of B+. "
            "Planning for Fall 2026. Max 15 credits. "
            "Can I take COP 2513 (Object Oriented Programming)?"
        ),
        "expected_decision": "Eligible",
        "expected_abstain": False,
        "grading_notes": (
            "COP 2513 prereq: COP 2510 minimum grade C. Student has B+ — satisfies it. "
            "1 pt for Eligible with citation of COP 2513 prerequisites. "
            "0 pt for any other decision."
        ),
    },
    {
        "id": 2,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "I have only completed COP 2510 (grade B), but I have not taken MAC 2281. "
            "Planning for Fall 2026. Max 15 credits. "
            "Can I take COP 3514 (Program Design)?"
        ),
        "expected_decision": "Not eligible",
        "expected_abstain": False,
        "grading_notes": (
            "COP 3514 requires COP 2510 min B AND MAC 2281 min B. "
            "Student has COP 2510 (B) but lacks MAC 2281 entirely. "
            "1 pt for Not eligible with citation of both prereqs for COP 3514. "
            "0.5 pt if system says Need more info asking about MAC 2281."
        ),
    },
    {
        "id": 3,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "I have completed COP 2510 (grade B) and MAC 2281 (grade B). "
            "Planning for Fall 2026. Max 15 credits. "
            "Can I take COP 3514 (Program Design)?"
        ),
        "expected_decision": "Eligible",
        "expected_abstain": False,
        "grading_notes": (
            "COP 3514 prereq: COP 2510 min B AND MAC 2281 min B. Both satisfied exactly. "
            "1 pt for Eligible with citations for both prereqs. "
            "NOTE: This is the fixed 'COMP2030 → COP 3514' query — use USF codes only."
        ),
    },
    {
        "id": 4,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "I completed COP 2510 with a grade of D. "
            "Planning for Fall 2026. Max 15 credits. "
            "Can I use COP 2510 as the prerequisite for COP 2513 (Object Oriented Programming)?"
        ),
        "expected_decision": "Not eligible",
        "expected_abstain": False,
        "grading_notes": (
            "COP 2513 requires COP 2510 minimum grade C. D is below C. "
            "1 pt for Not eligible with citation of the minimum grade requirement. "
            "0.5 pt if system says Need more info asking to confirm grade policy. "
            "0 pt for Eligible."
        ),
    },
    {
        "id": 5,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses list is: COP 2513 (grade A). That is all I have done. "
            "Planning for Fall 2026. Max 15 credits. "
            "Can I take COP 4538 (Data Structures and Algorithms)?"
        ),
        "expected_decision": "Not eligible",
        "expected_abstain": False,
        "grading_notes": (
            "COP 4538 requires COP 3515 min C. Student only has COP 2513 — missing COP 3515. "
            "1 pt for Not eligible with citation of COP 4538 → COP 3515 requirement. "
            "0 pt for Eligible."
        ),
    },
    {
        "id": 6,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "I have completed MAC 2281 (Calculus I) with a grade of A. "
            "Planning for Spring 2027. Max 12 credits. "
            "Can I take COT 3100 (Introduction to Discrete Structures)?"
        ),
        "expected_decision": "Eligible",
        "expected_abstain": False,
        "grading_notes": (
            "COT 3100 prereq: MAC 2281. Student has it with grade A. "
            "1 pt for Eligible with citation of COT 3100 prerequisites. "
            "0 pt for any other decision."
        ),
    },
    {
        "id": 7,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: none yet. Planning for Fall 2026. Max 15 credits. "
            "I have instructor consent from my professor for COP 2513. "
            "Can I skip the COP 2510 prerequisite for COP 2513?"
        ),
        "expected_decision": "Need more info",
        "expected_abstain": False,
        "grading_notes": (
            "If catalog documents an instructor-consent override policy, cite it. "
            "COP 3514 has 'Consent of Instructor or Department Permit Required' restriction — "
            "system should note this but consent details for COP 2513 may not be in catalog. "
            "1 pt for Need more info + citation of any consent/restriction policy found. "
            "0.5 pt for abstention noting consent policy exists but specifics not in catalog."
        ),
    },
    {
        "id": 8,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "I have completed COP 2513 with a grade of C. "
            "Planning for Fall 2026. Max 15 credits. "
            "Can I take COP 3515 (Advanced Program Design)?"
        ),
        "expected_decision": "Not eligible",
        "expected_abstain": False,
        "grading_notes": (
            "COP 3515 requires COP 2513 minimum grade B. Student has C — fails the minimum. "
            "1 pt for Not eligible with citation of the B minimum for COP 2513 in COP 3515 prereqs. "
            "0 pt for Eligible."
        ),
    },
    {
        "id": 9,
        "category": 1,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "I have completed COP 3515 (Advanced Program Design) with a grade of A. "
            "Planning for Fall 2026. Max 15 credits. "
            "Am I eligible to take COP 4538 (Data Structures and Algorithms)?"
        ),
        "expected_decision": "Eligible",
        "expected_abstain": False,
        "grading_notes": (
            "COP 4538 prereq: COP 3515 min C. Student has A — satisfies it. "
            "1 pt for Eligible with citation of COP 4538 prerequisites. "
            "0 pt for Not eligible."
        ),
    },
    {
        "id": 10,
        "category": 1,
        "query": (
            "I am a first-semester freshman enrolled in BS Computer Science at USF. "
            "I have completed zero courses. "
            "Planning for Fall 2026. Max 15 credits. "
            "Can I take CEN 4020 (Software Engineering)?"
        ),
        "expected_decision": "Not eligible",
        "expected_abstain": False,
        "grading_notes": (
            "CEN 4020 has capstone attribute and requires COP 4530 min C. "
            "Freshman has zero courses — clearly not eligible. "
            "1 pt for Not eligible with citation of CEN 4020 prerequisites. "
            "0 pt for Eligible."
        ),
    },

    # ── Category 2: Prerequisite chain questions (5) ──────────────────────────

    {
        "id": 11,
        "category": 2,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: none. Planning for Fall 2026. Max 15 credits. "
            "What is the full prerequisite chain I need to complete before I can take "
            "COP 4538 (Data Structures and Algorithms)?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "Full chain: COP 2510 → COP 2513 → COP 3515 → COP 4538. "
            "COP 3515 also needs MAC 1147, MAD2104, PHY 2020, STA 2023 all at min B. "
            "1 pt: all hops cited correctly with sources. "
            "0.5 pt: main COP chain correct, math/science co-reqs missed."
        ),
    },
    {
        "id": 12,
        "category": 2,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: none. Planning for Fall 2026. Max 15 credits. "
            "What is the minimum number of terms to reach COP 4703 (Advanced Database Systems), "
            "and what courses do I need in order?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "Chain: COP 2510 → COP 2513 → COP 3515 → COP 4538 → COP 4703 (4 sequential prereq steps). "
            "Term scheduling is NOT in the catalog — system should flag this. "
            "1 pt: correct full chain cited at every hop. "
            "0.5 pt: right chain, partial citations or missing the scheduling abstention."
        ),
    },
    {
        "id": 13,
        "category": 2,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: none. Planning for Fall 2026. Max 15 credits. "
            "What is the full list of courses I must complete before I can take "
            "COT 4400 (Analysis of Algorithms)?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "COT 4400 prereq: COP 4530. COP 4530 is NOT in our catalog documents. "
            "System should cite what it finds (COP 4530 as prereq for COT 4400) "
            "but must abstain on the COP 4530 chain since COP 4530 itself is not documented. "
            "1 pt: cites COT 4400 prereq correctly AND notes COP 4530 chain is not in catalog. "
            "0.5 pt: identifies COP 4530 requirement without flagging missing chain."
        ),
    },
    {
        "id": 14,
        "category": 2,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: COP 2510, MAC 2281. Planning for Fall 2026. Max 15 credits. "
            "Is COT 3100 (Introduction to Discrete Structures) listed as a required course "
            "in the BS Computer Science program requirements, "
            "and if so, which other courses depend on it?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "Cross-reference BS CS program page with COT 3100 page. "
            "1 pt: correct answer citing both the program requirements doc and COT 3100 prereq page. "
            "0.5 pt: correct answer with only one citation source."
        ),
    },
    {
        "id": 15,
        "category": 2,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: none. Planning for Fall 2026. Max 15 credits. "
            "What is the earliest term I could take COP 4703 (Advanced Database Systems), "
            "and what sequence of courses leads there?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "Chain: COP 2510 → COP 2513 → COP 3515 → COP 4538 → COP 4703. "
            "Course availability by term is NOT in catalog — must flag this. "
            "1 pt: correct cited sequence + abstention on exact term availability. "
            "0.5 pt: right sequence, no abstention on scheduling. "
            "0 pt: states a specific term without catalog support."
        ),
    },

    # ── Category 3: Program requirement questions (5) ─────────────────────────

    {
        "id": 16,
        "category": 3,
        "query": (
            "I am enrolled in BS Computer Science at USF, catalog year 2025-2026. "
            "My completed courses: COP 2510. Planning for Fall 2026. Max 15 credits. "
            "How many total credit hours are required to earn the "
            "Bachelor of Science in Computer Science?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "Answer: 120 total degree hours (from Computer_Science_B_S_C_S program page). "
            "1 pt: correct number (120) with citation of the program requirements page. "
            "0 pt: no citation or wrong number."
        ),
    },
    {
        "id": 17,
        "category": 3,
        "query": (
            "I am enrolled in BS Computer Science at USF, catalog year 2025-2026. "
            "My completed courses: COP 2510. Planning for Fall 2026. Max 15 credits. "
            "What are all the required core courses for the BS in Computer Science?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "Full list from Computer_Science_B_S_C_S program requirements page. "
            "1 pt: complete list with citation of the program page. "
            "0.5 pt: partial list or list without citation."
        ),
    },
    {
        "id": 18,
        "category": 3,
        "query": (
            "I am enrolled in BS Computer Science at USF, catalog year 2025-2026. "
            "My completed courses: COP 2510. Planning for Fall 2026. Max 15 credits. "
            "How many elective credits can I take from outside the Computer Science department "
            "and still have them count toward my BS CS degree?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "Elective rules from the BS CS program requirements page. "
            "1 pt: specific credit count with citation. "
            "0.5 pt: general reference to electives without a specific number."
        ),
    },
    {
        "id": 19,
        "category": 3,
        "query": (
            "I am enrolled in BS Computer Science at USF, catalog year 2025-2026. "
            "My completed courses: MAC 2281. Planning for Fall 2026. Max 15 credits. "
            "Can MAC 2281 count toward my BS Computer Science degree requirements?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "Cross-reference BS CS program requirements page for math/gen-ed requirements. "
            "1 pt: clear yes/no with citation from the program page. "
            "0.5 pt: ambiguous answer or missing citation."
        ),
    },
    {
        "id": 20,
        "category": 3,
        "query": (
            "I am enrolled in BS Computer Science at USF, catalog year 2025-2026. "
            "My completed courses: COP 2510. Planning for Fall 2026. Max 15 credits. "
            "What is the minimum GPA required to remain in good academic standing "
            "in the Computer Science program?"
        ),
        "expected_decision": None,
        "expected_abstain": False,
        "grading_notes": (
            "From Academic_Policies_and_Procedures or BS CS program page. "
            "1 pt: specific GPA value with citation. "
            "0.5 pt: general reference to academic standing without a specific GPA number."
        ),
    },

    # ── Category 4: Not-in-docs / trick questions (5) ────────────────────────

    {
        "id": 21,
        "category": 4,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: COP 2510. Planning for Fall 2026. Max 15 credits. "
            "Is COP 4710 (Database Design) being offered in Spring 2026?"
        ),
        "expected_decision": None,
        "expected_abstain": True,
        "grading_notes": (
            "Course scheduling is NOT in the catalog. "
            "1 pt: exact abstention phrase + suggestion to check the USF registrar or course schedule. "
            "0 pt: any attempt to answer whether the course is offered."
        ),
    },
    {
        "id": 22,
        "category": 4,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: COP 2510. Planning for Fall 2026. Max 15 credits. "
            "Who is the professor teaching COT 4400 (Analysis of Algorithms) next semester?"
        ),
        "expected_decision": None,
        "expected_abstain": True,
        "grading_notes": (
            "Professor / instructor assignments are NOT in the catalog. "
            "1 pt: abstention phrase + suggest department website or course schedule. "
            "0 pt: any name or guess provided."
        ),
    },
    {
        "id": 23,
        "category": 4,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: none. Planning for Fall 2026. Max 15 credits. "
            "I have 5 years of professional software engineering experience. "
            "Can I get a waiver for COP 2510 based on my industry experience?"
        ),
        "expected_decision": None,
        "expected_abstain": True,
        "grading_notes": (
            "Experience-based prerequisite waivers are NOT documented in the catalog. "
            "1 pt: abstention phrase + suggest contacting academic advisor. "
            "0.5 pt: abstention without advisor suggestion."
        ),
    },
    {
        "id": 24,
        "category": 4,
        "query": (
            "I am enrolled in BS Computer Science at USF. "
            "My completed courses: COP 2510, COP 2513. Planning for Fall 2026. Max 15 credits. "
            "What is the waitlist policy for overenrolled courses at USF?"
        ),
        "expected_decision": None,
        "expected_abstain": True,
        "grading_notes": (
            "Waitlist/overenrollment policy is NOT in the catalog documents we have. "
            "1 pt: abstention phrase + suggest the USF registrar or department. "
            "0 pt: any specific waitlist procedure stated without catalog support."
        ),
    },
    {
        "id": 25,
        "category": 4,
        "query": (
            "I am enrolled in BS Computer Science at USF with a cumulative GPA of 3.8. "
            "My completed courses: COP 2510, COP 2513. Planning for Fall 2026. Max 15 credits. "
            "Is there a scholarship specifically for Computer Science majors with a GPA above 3.5?"
        ),
        "expected_decision": None,
        "expected_abstain": True,
        "grading_notes": (
            "Scholarship details are NOT in the catalog. "
            "1 pt: abstention phrase + suggest USF financial aid office. "
            "0 pt: any scholarship claim made without catalog support."
        ),
    },
]

# ── Plan generation query (for transcript_2) — not in the 25-query eval set ───
# Used by evaluator.py to capture the course-plan transcript.
PLAN_TRANSCRIPT_QUERY = (
    "I am enrolled in BS Computer Science at USF, catalog year 2025-2026. "
    "Completed courses: COP 2510 (B+), COP 2513 (A), MAC 2281 (B+), COT 3100 (A). "
    "Planning for Fall 2026. Max 15 credits. "
    "Please suggest a full course plan for my next semester, "
    "with justification for each course based on prerequisites and degree requirements."
)
