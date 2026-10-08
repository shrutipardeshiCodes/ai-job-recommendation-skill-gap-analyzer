"""Quick sanity tests for all src/ modules."""
import sys
sys.path.insert(0, '.')

from src.preprocessing import clean_text
from src.skill_extraction import extract_skills, SKILL_VOCABULARY
from src.skill_gap import analyse_skill_gap
import pandas as pd

# ─── Test 1: clean_text ──────────────────────────────────────────────────────
t = clean_text('  Hello, WORLD!!! @#$  ')
assert 'hello' in t and 'world' in t, f'FAIL clean_text: {t}'
print(f'PASS clean_text  → {repr(t)}')

# ─── Test 2: extract_skills ───────────────────────────────────────────────────
skills = extract_skills('We need Python, SQL, Docker and FastAPI')
assert 'python' in skills, 'FAIL: python'
assert 'sql' in skills, 'FAIL: sql'
assert 'docker' in skills, 'FAIL: docker'
assert 'fastapi' in skills, 'FAIL: fastapi'
print(f'PASS extract_skills → {skills}')

# ─── Test 3: vocab size ────────────────────────────────────────────────────────
assert len(SKILL_VOCABULARY) > 50, 'FAIL: vocab too small'
print(f'PASS vocab size → {len(SKILL_VOCABULARY)} skills')

# ─── Test 4: analyse_skill_gap ────────────────────────────────────────────────
gap = analyse_skill_gap(
    ['python', 'sql', 'git'],
    ['python', 'sql', 'git', 'fastapi', 'docker']
)
assert 'fastapi' in gap['missing'], f'FAIL gap missing: {gap}'
assert gap['match_pct'] == 60.0, f'FAIL gap pct: {gap["match_pct"]}'
print(f'PASS analyse_skill_gap → match_pct={gap["match_pct"]}%')

# ─── Test 5: CSV exists ────────────────────────────────────────────────────────
df = pd.read_csv('data/jobs.csv', nrows=5, low_memory=False)
assert 'job_title' in df.columns, f'FAIL: job_title not in columns: {list(df.columns)}'
assert 'company_name' in df.columns, 'FAIL: company_name not in columns'
print(f'PASS CSV columns → {list(df.columns)}')

print()
print('=== ALL TESTS PASSED ===')
