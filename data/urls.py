"""
Master list of all catalog URLs to scrape.

Source: University of Massachusetts Lowell — Academic Catalog
Base  : https://www.uml.edu/catalog/
Date accessed: 2026-03-29

URL VERIFICATION NOTE
---------------------
UML uses CourseLeaf CIM for catalog management. If any URL below returns a 404,
visit https://www.uml.edu/catalog/ directly, navigate to the course, and copy
the updated URL into the matching entry below. The scraper logs all failures.
"""

from __future__ import annotations

from typing import List, TypedDict


class CatalogEntry(TypedDict):
    doc_id: str
    title: str
    url: str
    doc_type: str   # "course" | "program" | "policy"


# ── Course pages (20) ─────────────────────────────────────────────────────────
COURSE_URLS: List[CatalogEntry] = [
    {
        "doc_id": "COMP1010",
        "title": "COMP 1010 – Computing I",
        "url": "https://www.uml.edu/catalog/courses/comp/1010/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP1020",
        "title": "COMP 1020 – Computing II",
        "url": "https://www.uml.edu/catalog/courses/comp/1020/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP2010",
        "title": "COMP 2010 – Computing III",
        "url": "https://www.uml.edu/catalog/courses/comp/2010/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP2030",
        "title": "COMP 2030 – Discrete Structures",
        "url": "https://www.uml.edu/catalog/courses/comp/2030/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP3010",
        "title": "COMP 3010 – Organization of Programming Languages",
        "url": "https://www.uml.edu/catalog/courses/comp/3010/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP3040",
        "title": "COMP 3040 – Algorithms",
        "url": "https://www.uml.edu/catalog/courses/comp/3040/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP3050",
        "title": "COMP 3050 – Computer Architecture",
        "url": "https://www.uml.edu/catalog/courses/comp/3050/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP3080",
        "title": "COMP 3080 – Operating Systems",
        "url": "https://www.uml.edu/catalog/courses/comp/3080/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP4040",
        "title": "COMP 4040 – Theory of Computation",
        "url": "https://www.uml.edu/catalog/courses/comp/4040/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP4080",
        "title": "COMP 4080 – Computer Networks",
        "url": "https://www.uml.edu/catalog/courses/comp/4080/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP4110",
        "title": "COMP 4110 – Database Systems",
        "url": "https://www.uml.edu/catalog/courses/comp/4110/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP4130",
        "title": "COMP 4130 – Machine Learning",
        "url": "https://www.uml.edu/catalog/courses/comp/4130/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP4350",
        "title": "COMP 4350 – Software Engineering",
        "url": "https://www.uml.edu/catalog/courses/comp/4350/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP4610",
        "title": "COMP 4610 – GUI Programming I",
        "url": "https://www.uml.edu/catalog/courses/comp/4610/",
        "doc_type": "course",
    },
    {
        "doc_id": "COMP4960",
        "title": "COMP 4960 – Research / Senior Capstone",
        "url": "https://www.uml.edu/catalog/courses/comp/4960/",
        "doc_type": "course",
    },
    {
        "doc_id": "MATH1310",
        "title": "MATH 1310 – Calculus I",
        "url": "https://www.uml.edu/catalog/courses/math/1310/",
        "doc_type": "course",
    },
    {
        "doc_id": "MATH1320",
        "title": "MATH 1320 – Calculus II",
        "url": "https://www.uml.edu/catalog/courses/math/1320/",
        "doc_type": "course",
    },
    {
        "doc_id": "MATH2310",
        "title": "MATH 2310 – Calculus III",
        "url": "https://www.uml.edu/catalog/courses/math/2310/",
        "doc_type": "course",
    },
    {
        "doc_id": "MATH2340",
        "title": "MATH 2340 – Differential Equations",
        "url": "https://www.uml.edu/catalog/courses/math/2340/",
        "doc_type": "course",
    },
    {
        "doc_id": "MATH3220",
        "title": "MATH 3220 – Statistics",
        "url": "https://www.uml.edu/catalog/courses/math/3220/",
        "doc_type": "course",
    },
]

# ── Program / degree requirement pages (3) ────────────────────────────────────
PROGRAM_URLS: List[CatalogEntry] = [
    {
        "doc_id": "BSCS_REQUIREMENTS",
        "title": "BS Computer Science – Degree Requirements",
        "url": "https://www.uml.edu/catalog/undergraduate/sciences/computer-science/bs-computer-science/",
        "doc_type": "program",
    },
    {
        "doc_id": "CS_MINOR",
        "title": "Computer Science Minor – Requirements",
        "url": "https://www.uml.edu/catalog/undergraduate/sciences/computer-science/cs-minor/",
        "doc_type": "program",
    },
    {
        "doc_id": "CS_CONCENTRATION",
        "title": "CS Concentrations / Tracks (AI, Systems, Security)",
        "url": "https://www.uml.edu/catalog/undergraduate/sciences/computer-science/",
        "doc_type": "program",
    },
]

# ── Academic policy pages (2) ────────────────────────────────────────────────
POLICY_URLS: List[CatalogEntry] = [
    {
        "doc_id": "POLICY_GRADING",
        "title": "Academic Grading Policy – Grading System, Repeats, Minimum Grades",
        "url": "https://www.uml.edu/catalog/policies/academic/grading-system/",
        "doc_type": "policy",
    },
    {
        "doc_id": "POLICY_CREDITS",
        "title": "Academic Policy – Credit Limits, Residency, Transfer Credits",
        "url": "https://www.uml.edu/catalog/policies/academic/credit-hour/",
        "doc_type": "policy",
    },
]

# ── Combined list (all 25) ────────────────────────────────────────────────────
ALL_URLS: List[CatalogEntry] = COURSE_URLS + PROGRAM_URLS + POLICY_URLS
