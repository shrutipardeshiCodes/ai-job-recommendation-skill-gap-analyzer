"""
skill_extraction.py
-------------------
Rule-based skill extraction from text.

How it works:
  1. A predefined vocabulary of common tech/data skills is defined.
  2. Each skill is searched (as a whole word / phrase) in the text.
  3. Skills that are found are returned as a list.

This is intentionally simple and explainable — no machine learning here.
The purpose is to convert free-form text into a structured skill set.
"""

import re

# ─────────────────────────────────────────────────────────────
# SKILL VOCABULARY
# Add or remove skills here. Keep it realistic.
# ─────────────────────────────────────────────────────────────
SKILL_VOCABULARY = [
    # Programming Languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript",
    "r", "go", "rust", "kotlin", "swift", "php", "ruby", "scala", "matlab",

    # Web / Frontend
    "html", "css", "react", "angular", "vue", "next.js", "bootstrap",
    "tailwind", "jquery",

    # Backend / Frameworks
    "node.js", "django", "flask", "fastapi", "spring", "express", "laravel",
    "asp.net", "rest api", "graphql",

    # Databases
    "sql", "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis",
    "cassandra", "elasticsearch", "firebase",

    # Data & ML
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data analysis", "data science", "statistics",
    "tensorflow", "pytorch", "keras", "scikit-learn", "opencv",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",

    # BI / Analytics
    "excel", "power bi", "tableau", "looker", "qlik", "sas", "spss",

    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes",
    "ci/cd", "jenkins", "github actions", "terraform", "ansible",

    # Version Control / Tools
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "agile", "scrum",

    # Other
    "linux", "bash", "shell scripting", "api", "microservices",
    "spark", "hadoop", "kafka", "airflow", "dbt", "snowflake",
    "communication", "problem solving", "teamwork", "leadership",
]

# Sort by length descending so multi-word skills match before sub-strings
SKILL_VOCABULARY.sort(key=len, reverse=True)


def extract_skills(text: str) -> list[str]:
    """
    Extract skills from a text string using rule-based matching.

    Parameters
    ----------
    text : any free-form text (job description, user input, etc.)

    Returns
    -------
    List of skill strings that were found in the text.
    """
    if not isinstance(text, str) or text.strip() == "":
        return []

    text_lower = text.lower()
    found = []

    for skill in SKILL_VOCABULARY:
        # Use word-boundary regex so 'r' doesn't match inside 'error'
        pattern = r"(?<![a-zA-Z0-9\+\#])" + re.escape(skill) + r"(?![a-zA-Z0-9\+\#])"
        if re.search(pattern, text_lower):
            found.append(skill)

    return found


def skills_list_to_text(skills: list[str]) -> str:
    """
    Join a list of skills into a single space-separated string.
    Used to create a user-profile text for TF-IDF vectorisation.
    """
    return " ".join(skills)


def get_all_skills() -> list[str]:
    """Return the full skill vocabulary (sorted alphabetically for display)."""
    return sorted(SKILL_VOCABULARY)
