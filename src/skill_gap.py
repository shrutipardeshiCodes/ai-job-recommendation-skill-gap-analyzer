"""
skill_gap.py
------------
Skill Gap Analysis and Priority Recommendation.

How it works (viva-ready explanation):
  1. Extract skills required by each recommended job.
  2. Compare with skills the user already has.
  3. Find:
       - Matching skills  = user_skills ∩ job_skills
       - Missing skills   = job_skills  − user_skills
  4. Across ALL recommended jobs, count how often each missing
     skill appears → frequency table.
  5. Rank missing skills by frequency → Priority Recommendation.
     Higher frequency = more important to learn first.
"""

from collections import Counter

from src.skill_extraction import extract_skills


def analyse_skill_gap(
    user_skills: list[str],
    job_skills: list[str],
) -> dict:
    """
    Compare user skills against a single job's required skills.

    Parameters
    ----------
    user_skills : skills the user already has
    job_skills  : skills required by the job

    Returns
    -------
    dict with keys:
        matching (list) — skills the user has that the job wants
        missing  (list) — skills the job wants that the user lacks
        match_pct (float) — % of required skills the user already has
    """
    user_set = set(s.lower() for s in user_skills)
    job_set  = set(s.lower() for s in job_skills)

    matching = sorted(user_set & job_set)
    missing  = sorted(job_set  - user_set)

    match_pct = (
        round(len(matching) / len(job_set) * 100, 1) if job_set else 0.0
    )

    return {
        "matching" : matching,
        "missing"  : missing,
        "match_pct": match_pct,
    }


def priority_skills(
    recommended_df,
    user_skills: list[str],
    top_n: int = 10,
) -> list[dict]:
    """
    Identify and rank missing skills across ALL recommended jobs.

    Priority is determined by frequency:
      → A skill that appears as missing in 8 out of 10 jobs is more
        important than one missing in only 2 jobs.

    Parameters
    ----------
    recommended_df : DataFrame returned by JobRecommender.recommend()
    user_skills    : list of user's current skills
    top_n          : number of priority skills to return

    Returns
    -------
    List of dicts: [{"skill": ..., "frequency": ..., "priority": ...}, ...]
    """
    user_set = set(s.lower() for s in user_skills)
    all_missing = []

    for _, row in recommended_df.iterrows():
        job_s = row.get("job_skills", [])
        if isinstance(job_s, str):
            job_s = extract_skills(job_s)
        job_set = set(s.lower() for s in job_s)
        missing = job_set - user_set
        all_missing.extend(missing)

    if not all_missing:
        return []

    counts = Counter(all_missing)
    top_missing = counts.most_common(top_n)

    return [
        {
            "skill"    : skill,
            "frequency": freq,
            "priority" : rank + 1,
        }
        for rank, (skill, freq) in enumerate(top_missing)
    ]


def format_skill_gap_report(gap: dict) -> str:
    """
    Return a human-readable text summary of a skill gap result.

    Parameters
    ----------
    gap : dict returned by analyse_skill_gap()
    """
    lines = []

    if gap["matching"]:
        lines.append("✅ **Skills you already have:**")
        for s in gap["matching"]:
            lines.append(f"   ✓ {s.title()}")
    else:
        lines.append("⚠️  No matching skills found for this job.")

    lines.append("")

    if gap["missing"]:
        lines.append("⚠️ **Skills you need to learn:**")
        for s in gap["missing"]:
            lines.append(f"   ✗ {s.title()}")
    else:
        lines.append("🎉 You already have all required skills for this job!")

    lines.append(f"\n📊 **Skill match:** {gap['match_pct']}% of required skills covered")

    return "\n".join(lines)
