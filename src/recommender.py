"""
recommender.py
--------------
Content-based Job Recommendation using TF-IDF + Cosine Similarity.

ML Workflow (easy to explain in viva):
  ┌──────────────────────────────────────────────────────────┐
  │  User profile text                                       │
  │  (skills + preferred role + experience)                  │
  │            │                                             │
  │            ▼                                             │
  │  TF-IDF Vectorisation                                    │
  │  • TF  = term frequency in a document                   │
  │  • IDF = log(N / df)  — rare terms get higher weight    │
  │  • Result: a numeric vector for each job + user profile │
  │            │                                             │
  │            ▼                                             │
  │  Cosine Similarity                                       │
  │  • cos θ = (A · B) / (|A| × |B|)                       │
  │  • Score ranges 0 (no match) → 1 (perfect match)        │
  │            │                                             │
  │            ▼                                             │
  │  Ranked job recommendations (top-K)                      │
  └──────────────────────────────────────────────────────────┘
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.skill_extraction import extract_skills


class JobRecommender:
    """
    TF-IDF based content-filtering recommender.

    Usage
    -----
    recommender = JobRecommender()
    recommender.fit(df)           # build TF-IDF matrix from job data
    results = recommender.recommend(user_text, top_k=10)
    """

    def __init__(self):
        # TfidfVectorizer settings — kept simple for explainability
        self.vectorizer = TfidfVectorizer(
            max_features=5000,   # top 5000 most informative terms
            ngram_range=(1, 2),  # unigrams + bigrams (e.g. "machine learning")
            stop_words="english",
            min_df=2,            # ignore terms in fewer than 2 documents
        )
        self.tfidf_matrix = None
        self.df = None

    def fit(self, df: pd.DataFrame):
        """
        Build the TF-IDF matrix from the jobs dataset.

        Parameters
        ----------
        df : preprocessed DataFrame with a 'combined_text' column
        """
        self.df = df.copy().reset_index(drop=True)
        print("[Recommender] Fitting TF-IDF on job corpus ...")
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["combined_text"].fillna("")
        )
        print(f"  TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print("[Recommender] TF-IDF fitted.")

    def recommend(
        self,
        user_text: str,
        top_k: int = 10,
        preferred_location: str = "",
    ) -> pd.DataFrame:
        """
        Recommend top-K jobs for a given user profile text.

        Parameters
        ----------
        user_text          : free-form text describing the user's skills/role
        top_k              : number of top jobs to return
        preferred_location : optional location filter (soft — jobs from this
                             location get a small boost, not a hard filter)

        Returns
        -------
        DataFrame with columns:
          title, company, location, match_score, matching_skills, job_skills
        """
        if self.tfidf_matrix is None:
            raise RuntimeError("Call fit() before recommend().")

        # ── Step 1: Vectorise user profile ───────────────────────────────────
        user_vec = self.vectorizer.transform([user_text])

        # ── Step 2: Compute cosine similarity ────────────────────────────────
        similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()

        # ── Step 3: Location boost (soft preference, not hard filter) ────────
        if preferred_location.strip():
            loc_lower = preferred_location.lower()
            location_boost = self.df["location"].str.contains(
                loc_lower, na=False
            ).astype(float) * 0.05   # 5% boost for matching location
            similarities = similarities + location_boost

        # ── Step 4: Rank by similarity score ─────────────────────────────────
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = self.df.iloc[top_indices].copy()
        results["match_score"] = (similarities[top_indices] * 100).round(2)

        # ── Step 5: Identify matching skills per job ──────────────────────────
        user_skills = set(extract_skills(user_text))
        results["job_skills"] = results["combined_text"].apply(extract_skills)
        results["matching_skills"] = results["job_skills"].apply(
            lambda job_s: sorted(user_skills & set(job_s))
        )

        # Return only relevant columns
        out_cols = [
            "title", "company", "location",
            "match_score", "matching_skills", "job_skills",
        ]
        # Keep category if available
        if "category" in results.columns:
            out_cols.insert(3, "category")

        return results[out_cols].reset_index(drop=True)
