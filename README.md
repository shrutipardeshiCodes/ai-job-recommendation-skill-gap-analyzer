# 🎯 AI Job Recommendation & Skill Gap Analyzer

> **SDG 8 — Decent Work and Economic Growth**  
> Helping students and freshers find suitable job opportunities and identify the skills they need to develop.

---

## 📋 Table of Contents
1. [Problem Statement](#problem-statement)
2. [Proposed Solution](#proposed-solution)
3. [Features](#features)
4. [Dataset](#dataset)
5. [ML Methodology](#ml-methodology)
6. [Technologies Used](#technologies-used)
7. [Project Structure](#project-structure)
8. [How to Run](#how-to-run)
9. [Evaluation](#evaluation)
10. [Limitations](#limitations)
11. [Future Improvements](#future-improvements)

---

## 🔴 Problem Statement

Many B.Tech/B.Sc students and freshers face two major challenges when entering the job market:

1. **Not knowing which jobs match their current skills** — they apply randomly and face rejections.
2. **Not knowing which skills to learn next** — they learn skills that may not be required for their target roles.

A targeted, data-driven tool can bridge this gap by analyzing the student's skills and recommending suitable jobs along with a clear skill development roadmap.

---

## ✅ Proposed Solution

An ML-powered web application that:

- Takes a student's current skills and job preferences as input.
- Recommends the top matching jobs from a real dataset.
- Identifies the exact skill gap between the student's profile and each job.
- Prioritizes which missing skills to learn first.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 Job Recommendation | Top-K job matches using TF-IDF + Cosine Similarity |
| 🧩 Skill Gap Analysis | Set-difference between user skills and job requirements |
| 📚 Priority Learning | Frequency-ranked missing skills across recommended jobs |
| 📊 Visualisations | Intuitive skill gap distribution and demand charts |
| 💻 Web Interface | Clean Streamlit UI — no installation required for end-users |

---

## 📂 Dataset

**Name:** Luke Barousse — Data Jobs 2023  
**Source:** [huggingface.co/datasets/lukebarousse/data_jobs](https://huggingface.co/datasets/lukebarousse/data_jobs)  
**Size:** ~785,000 job postings (we sample 3,000 for performance)  
**License:** Community / Open Access

### Key Columns Used

| Column | Description |
|--------|-------------|
| `job_title` | Title of the job posting |
| `company_name` | Hiring company |
| `job_location` | Job location |
| `job_skills` | Pre-extracted skill list (comma-separated) |
| `job_title_short` | Short category (e.g., Data Analyst, ML Engineer) |
| `job_schedule_type` | Full-time, Part-time, Contract, etc. |

---

## 🧠 ML Methodology

### Complete Workflow

```
Dataset (jobs.csv)
       │
       ▼
Data Cleaning
  ├─ Missing value handling (fillna)
  ├─ Duplicate removal (drop_duplicates)
  ├─ Text normalization (lowercase, strip special chars)
  └─ Column standardization (rename to standard names)
       │
       ▼
Feature Extraction
  ├─ Combine: job_title + job_title_short + job_skills + job_description
  └─ Rule-based skill extraction from text (skill_extraction.py)
       │
       ▼
TF-IDF Vectorisation
  ├─ TF = Term Frequency = count(term in doc) / total terms in doc
  ├─ IDF = Inverse Document Frequency = log(N / doc_freq)
  └─ TF-IDF = TF × IDF  →  numeric matrix (jobs × vocabulary)
       │
       ▼
Cosine Similarity
  ├─ User profile text → TF-IDF vector
  ├─ cos θ = (user_vec · job_vec) / (|user_vec| × |job_vec|)
  └─ Score: 0 = no match, 1 = perfect match
       │
       ▼
Job Recommendation (Top-K)
  └─ Rank jobs by similarity score, return top K
       │
       ▼
Skill Gap Analysis
  ├─ Job skills = extract_skills(job combined_text)
  ├─ Matching = user_skills ∩ job_skills
  └─ Missing  = job_skills  - user_skills
       │
       ▼
Personalized Skill Recommendation
  └─ Count frequency of each missing skill across all top-K jobs
     → Higher frequency = higher priority to learn
```

---

### How TF-IDF Works in This Project

**TF-IDF** (Term Frequency–Inverse Document Frequency) converts text into numbers.

- **Term Frequency (TF):** How often a word appears in a single document (job).
  ```
  TF("python", job_1) = count("python" in job_1) / total words in job_1
  ```
- **Inverse Document Frequency (IDF):** Penalizes common words that appear in almost every job (like "the", "and"), and rewards rare, meaningful words.
  ```
  IDF("python") = log( total_jobs / jobs_containing_"python" )
  ```
- **Final TF-IDF score:**
  ```
  TF-IDF = TF × IDF
  ```
- The result is a **matrix** where each row is a job and each column is a word. Common words get low values; important skills like "tensorflow" or "postgresql" get high values.

---

### How Cosine Similarity Works

Cosine Similarity measures the **angle** between two vectors:

```
cos θ = (A · B) / (|A| × |B|)
```

- `A` = TF-IDF vector of the user's profile
- `B` = TF-IDF vector of a job
- If θ = 0° → cos θ = 1.0 (perfect match)
- If θ = 90° → cos θ = 0.0 (no match)

Using angle instead of distance means the **length of the text doesn't matter** — only the direction (i.e., the topics) matters.

---

### How Skill Gap Analysis Works

```python
user_skills = {"python", "sql", "git"}
job_skills  = {"python", "sql", "git", "fastapi", "postgresql", "docker"}

matching = user_skills ∩ job_skills = {"python", "sql", "git"}
missing  = job_skills - user_skills  = {"fastapi", "postgresql", "docker"}

match_pct = len(matching) / len(job_skills) × 100 = 50%
```

This is pure **set theory** — no black-box models. Easy to explain and verify.

---

## 🛠 Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core programming language |
| Pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF vectorization, Cosine Similarity |
| Streamlit | Web interface |
| Plotly | Interactive charts |
| Matplotlib | Static plots (in notebook) |
| Hugging Face Datasets | Free dataset download |

---

## 📁 Project Structure

```
job recommendation/
│
├── app.py                    ← Main Streamlit application
├── download_data.py          ← Script to download the dataset
├── requirements.txt          ← Python dependencies
├── README.md                 ← This file
│
├── data/
│   └── jobs.csv              ← Dataset (download using download_data.py)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      ← Data loading & cleaning
│   ├── skill_extraction.py   ← Rule-based skill extraction
│   ├── recommender.py        ← TF-IDF + Cosine Similarity recommender
│   └── skill_gap.py          ← Skill gap & priority recommendation
│
├── notebooks/
│   └── analysis.ipynb        ← Exploratory analysis & ML walkthrough
│
└── model/                    ← (Reserved for saving fitted models)
```

---

## 🚀 How to Run

### Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2 — Download the Dataset

```bash
python download_data.py
```

This downloads ~230 MB from Hugging Face (free, no login required).  
The file is saved as `data/jobs.csv`.

### Step 3 — Run the Application

```bash
streamlit run app.py
```

Open your browser to: **http://localhost:8501**

---

## 📊 Evaluation

This is a **recommendation system**, not a classification problem.  
Traditional accuracy is not applicable. We use:

| Metric | Description |
|--------|-------------|
| **Cosine Similarity Score** | Direct measure of profile ↔ job relevance (0–1) |
| **Match % displayed** | Similarity × 100, shown per job |
| **Manual Inspection** | Check if top jobs logically match the input profile |

> **We do NOT fabricate accuracy numbers.**  
> The system is evaluated honestly based on similarity scores and manual verification.

---

## ⚠️ Limitations

1. **Dataset staleness** — The dataset is from 2023. Job requirements evolve.
2. **Dataset quality** — Recommendations are only as good as the data.
3. **Skill extraction** — Rule-based matching may miss misspelled or very new skills.
4. **Similarity ≠ Hiring** — A high match score does not guarantee a job offer.
5. **No personalization** — The system does not learn from user feedback.
6. **English only** — The system works only with English text.

---

## 🔮 Future Improvements

1. **Larger dataset** — Include more job sources (Naukri, LinkedIn, Indeed).
2. **Better skill extraction** — Use NER (Named Entity Recognition) models.
3. **User feedback loop** — Allow users to rate recommendations for improvement.
4. **Resume upload** — Parse a PDF resume to auto-extract user skills.
5. **Learning path links** — Link each missing skill to free learning resources (Coursera, YouTube).
6. **Collaborative filtering** — Recommend jobs based on similar users' choices.

---

## 📌 SDG 8 Connection

**Sustainable Development Goal 8** targets:
> *"Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all."*

This project directly contributes by:
- Helping fresh graduates enter the workforce faster.
- Reducing the mismatch between job seekers and job requirements.
- Providing a free, accessible tool for career planning.

---

*Built for academic demonstration — B.Tech CSE Final Year Project*
