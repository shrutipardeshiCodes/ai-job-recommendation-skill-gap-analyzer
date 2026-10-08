"""
app.py — AI Job Recommendation & Skill Gap Analyzer
====================================================
Streamlit front-end.

Run with:
    streamlit run app.py
"""

import os
import sys

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── make src/ importable regardless of cwd ──────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.preprocessing    import load_and_preprocess
from src.skill_extraction import extract_skills, get_all_skills
from src.recommender      import JobRecommender
from src.skill_gap        import analyse_skill_gap, priority_skills

# ════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Job Recommendation & Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — clean, modern, distraction-free
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Deploy button & Streamlit header/footer clutter ── */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
.stDeployButton,
[data-testid="stDeployButton"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
#MainMenu,
footer {
    display: none !important;
    visibility: hidden !important;
}

/* ── Page container spacing ── */
.main .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
    margin: 0 auto;
}

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}

/* ── Header strip ── */
.header-strip {
    background: linear-gradient(90deg, #7f00ff, #e100ff);
    padding: 1.8rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(127, 0, 255, 0.25);
}
.header-strip h1 {
    color: white;
    font-size: 1.9rem;
    font-weight: 700;
    margin: 0;
}
.header-strip p {
    color: rgba(255,255,255,0.9);
    margin: 0.4rem 0 0 0;
    font-size: 0.95rem;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 600;
    color: #c084fc;
    border-left: 4px solid #7f00ff;
    padding-left: 0.75rem;
    margin: 1.8rem 0 1rem 0;
}

/* ── Job card ── */
.job-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.9rem;
    backdrop-filter: blur(6px);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.job-card:hover {
    transform: translateY(-2px);
    border-color: #a855f7;
}
.job-card h3 {
    margin: 0 0 0.3rem 0;
    color: #f0abfc;
    font-size: 1.1rem;
}
.job-card .meta {
    color: #a0a0c0;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}
.score-badge {
    display: inline-block;
    background: linear-gradient(90deg, #7f00ff, #e100ff);
    color: white;
    font-weight: 600;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.82rem;
}

/* ── Skill badges ── */
.skill-badge-match {
    display: inline-block;
    background: rgba(74, 222, 128, 0.15);
    border: 1px solid #4ade80;
    color: #4ade80;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 3px;
}
.skill-badge-miss {
    display: inline-block;
    background: rgba(251, 191, 36, 0.12);
    border: 1px solid #fbbf24;
    color: #fbbf24;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 3px;
}

/* ── Info box ── */
.info-box {
    background: rgba(127, 0, 255, 0.12);
    border: 1px solid rgba(127, 0, 255, 0.3);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.88rem;
    color: #e2e8f0;
}

/* ── Priority cards ── */
.priority-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.priority-num {
    background: linear-gradient(135deg, #7f00ff, #e100ff);
    color: white;
    font-weight: 700;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 0.85rem;
}

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 0.8rem 1rem;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (cached so it only runs once per session)
# ════════════════════════════════════════════════════════════════════════════
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "jobs.csv")


@st.cache_resource(show_spinner="⚙️ Loading and preparing dataset …")
def get_recommender():
    """Load data, fit TF-IDF recommender. Cached across Streamlit reruns."""
    if not os.path.exists(DATA_PATH):
        return None, None
    df = load_and_preprocess(DATA_PATH, sample_size=3000)
    rec = JobRecommender()
    rec.fit(df)
    return df, rec


df, recommender = get_recommender()

# ════════════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-strip">
  <h1>🎯 AI Job Recommendation & Skill Gap Analyzer</h1>
  <p>Helping students and freshers find matching job roles and identify skills to learn next</p>
</div>
""", unsafe_allow_html=True)

# Dataset not found guard
if df is None or recommender is None:
    st.error("""
    **Dataset not found at `data/jobs.csv`.**

    Please ensure `data/jobs.csv` is present in the project folder and restart Streamlit.
    """)
    st.stop()


# ════════════════════════════════════════════════════════════════════════════
# CANDIDATE PROFILE & SKILLS
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">👤 Candidate Profile & Skills</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    user_name = st.text_input("Your Name (optional)", placeholder="e.g. Priya Sharma")
    education = st.selectbox(
        "Current Education",
        ["B.Tech / B.E. (CS/IT)", "B.Tech / B.E. (Other)", "B.Sc (CS/IT)",
         "BCA", "MCA", "M.Tech", "MBA", "Diploma", "12th Pass / Fresher", "Other"],
    )
    experience = st.selectbox(
        "Experience Level",
        ["Fresher (0 years)", "0–1 years", "1–2 years", "2–4 years", "4+ years"],
    )

with col2:
    preferred_role = st.text_input(
        "Preferred Job Role",
        placeholder="e.g. Data Analyst, Python Developer, ML Engineer",
    )
    preferred_location = st.text_input(
        "Preferred Location",
        placeholder="e.g. Bangalore, Remote, Hyderabad, Mumbai",
    )
    top_k = st.slider("Number of job recommendations", 5, 15, 8)

# ── Skill selection ──────────────────────────────────────────────────────────
st.markdown("**Select your skills:**")
all_skills_sorted = get_all_skills()

selected_skills_multi = st.multiselect(
    "Choose from vocabulary (type to search):",
    options=all_skills_sorted,
    default=[],
)

custom_skills_text = st.text_input(
    "➕ Add any other skills (comma-separated):",
    placeholder="e.g. fastApi, seaborn, flutter",
)

# Combine multiselect + custom skills
custom_skills = [s.strip().lower() for s in custom_skills_text.split(",") if s.strip()]
user_skills   = list(set(selected_skills_multi + custom_skills))

if user_skills:
    st.markdown(f"**Selected Skills ({len(user_skills)}):**")
    badges = " ".join(
        f'<span class="skill-badge-match">{s.title()}</span>' for s in sorted(user_skills)
    )
    st.markdown(badges, unsafe_allow_html=True)

st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)

# ── Primary action button ────────────────────────────────────────────────────
analyze_btn = st.button("🚀 Find Matching Jobs", type="primary", use_container_width=True)

if analyze_btn:
    if not user_skills and not preferred_role.strip():
        st.warning("⚠️ Please select at least one skill or enter a preferred job role.")
        st.stop()

    # Build user profile text for TF-IDF
    user_profile_text = " ".join(user_skills) + " " + preferred_role
    if experience != "Fresher (0 years)":
        user_profile_text += " " + experience

    with st.spinner("🔄 Finding best matching jobs using TF-IDF & Cosine Similarity …"):
        results = recommender.recommend(
            user_text=user_profile_text,
            top_k=top_k,
            preferred_location=preferred_location,
        )

    # Save to session state so results persist across interaction
    st.session_state["results"]           = results
    st.session_state["user_skills"]       = user_skills
    st.session_state["user_profile_text"] = user_profile_text
    st.session_state["user_name"]         = user_name


# ════════════════════════════════════════════════════════════════════════════
# RESULTS
# ════════════════════════════════════════════════════════════════════════════
if "results" in st.session_state:
    results     = st.session_state["results"]
    user_skills = st.session_state["user_skills"]
    name_str    = st.session_state.get("user_name", "")
    greeting    = f"Hello **{name_str}**! " if name_str else ""

    st.success(
        f"{greeting}Found **{len(results)}** matching jobs for your profile."
    )

    # ── Summary metrics ──────────────────────────────────────────────────────
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Recommended Jobs", len(results))
    mcol2.metric("Highest Match",     f"{results['match_score'].max():.1f}%")
    mcol3.metric("Average Match",     f"{results['match_score'].mean():.1f}%")

    # ══════════════════════════════════════════════════════════════════════════
    # TOP JOB RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">💼 Top Recommended Jobs</div>', unsafe_allow_html=True)

    for i, row in results.iterrows():
        cat_tag = f" &nbsp;·&nbsp; 🏷️ {row['category'].title()}" if row.get('category', '') else ""
        match_badges = " ".join(
            f'<span class="skill-badge-match">{s.title()}</span>'
            for s in row["matching_skills"][:8]
        ) or "<i style='color:#888'>General profile match</i>"

        st.markdown(f"""
        <div class="job-card">
          <h3>#{i+1} &nbsp; {row['title'].title()}</h3>
          <div class="meta">
            🏢 {row['company'].title() or 'Company N/A'} &nbsp;·&nbsp;
            📍 {row['location'].title() or 'Location N/A'}
            {cat_tag}
          </div>
          <span class="score-badge">Match: {row['match_score']:.1f}%</span>
          <div style="margin-top:0.6rem; font-size:0.82rem; color:#a0a0c0;">Matching skills:</div>
          <div style="margin-top:0.3rem;">{match_badges}</div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SKILL GAP ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">🧩 Skill Gap Analysis</div>', unsafe_allow_html=True)

    job_options = [
        f"#{i+1} — {row['title'].title()} @ {row['company'].title()}"
        for i, row in results.iterrows()
    ]
    selected_job_label = st.selectbox("Select a job to inspect skill requirements:", job_options)
    selected_idx = job_options.index(selected_job_label)
    selected_job = results.iloc[selected_idx]

    gap = analyse_skill_gap(
        user_skills=user_skills,
        job_skills =selected_job["job_skills"],
    )

    gap_col1, gap_col2 = st.columns([3, 2])

    with gap_col1:
        st.markdown(f"**Skill Coverage: {gap['match_pct']}%**")
        st.progress(int(gap["match_pct"]))

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            st.markdown("**✅ Skills You Have:**")
            if gap["matching"]:
                for s in gap["matching"]:
                    st.markdown(
                        f'<span class="skill-badge-match">✓ {s.title()}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No matching skills detected for this specific job.")

        with sub_col2:
            st.markdown("**⚠️ Missing Skills to Learn:**")
            if gap["missing"]:
                for s in gap["missing"]:
                    st.markdown(
                        f'<span class="skill-badge-miss">✗ {s.title()}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.success("🎉 You already possess all key skills for this role!")

    with gap_col2:
        if gap["matching"] or gap["missing"]:
            fig_donut = go.Figure(go.Pie(
                labels=["Skills You Have", "Skills Missing"],
                values=[len(gap["matching"]), len(gap["missing"])],
                hole=0.55,
                marker_colors=["#4ade80", "#fbbf24"],
                textfont_size=12,
            ))
            fig_donut.update_layout(
                title=dict(text="Skill Distribution", font=dict(color="#e0e0e0", size=14)),
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e0e0e0",
                showlegend=True,
                height=260,
                margin=dict(l=10, r=10, t=35, b=10),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RECOMMENDED SKILLS TO LEARN NEXT
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">📚 Recommended Skills to Learn Next</div>', unsafe_allow_html=True)

    priorities = priority_skills(results, user_skills, top_n=6)

    if priorities:
        st.markdown(
            "<div class='info-box'>These skills appear most frequently across your recommended jobs. "
            "Learning these will give you the highest boost in job eligibility.</div>",
            unsafe_allow_html=True,
        )

        prio_col1, prio_col2 = st.columns([1, 1])

        with prio_col1:
            for item in priorities:
                st.markdown(f"""
                <div class="priority-card">
                  <div class="priority-num">{item['priority']}</div>
                  <div>
                    <b style="color:#f0abfc; font-size:1rem;">{item['skill'].title()}</b>
                    <br><span style="font-size:0.8rem;color:#a0a0c0">
                      Required in {item['frequency']} of your {len(results)} recommended jobs
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with prio_col2:
            prio_df = pd.DataFrame(priorities)
            fig_prio = px.bar(
                prio_df,
                x="frequency",
                y=prio_df["skill"].str.title(),
                orientation="h",
                color="frequency",
                color_continuous_scale="Purples",
                labels={"frequency": "Jobs Requiring Skill", "y": "Skill"},
                title="Skill Demand Across Matching Jobs",
            )
            fig_prio.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font_color   ="#e0e0e0",
                height       =280,
                margin       =dict(l=10, r=10, t=35, b=10),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_prio, use_container_width=True)

    else:
        st.success("🎉 You already possess all key skills for your recommended jobs!")

    # ── Clean footer note ─────────────────────────────────────────────────────
    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    💡 <b>Note:</b> This is an academic career-support tool powered by TF-IDF & Cosine Similarity.
    Recommendations help identify career directions and skill gaps.
    </div>
    """, unsafe_allow_html=True)

else:
    # Initial state before clicking button
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#6b7280;">
      <div style="font-size:3.5rem; margin-bottom: 0.5rem;">🎯</div>
      <h3 style="color:#c084fc; font-weight:500;">Fill in your details above and click <em>Find Matching Jobs</em></h3>
      <p style="font-size:0.95rem; max-width:600px; margin: 0.5rem auto;">
        Get matching job recommendations, discover your skill gaps, and see which skills to prioritize learning next.
      </p>
    </div>
    """, unsafe_allow_html=True)
