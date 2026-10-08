"""End-to-end test for the recommender pipeline."""
import sys
sys.path.insert(0, '.')

from src.preprocessing import load_and_preprocess
from src.recommender import JobRecommender
from src.skill_gap import priority_skills

print('=== End-to-End Recommender Test ===')
df = load_and_preprocess('data/jobs.csv', sample_size=1000)
print('Preprocessing OK, shape:', df.shape)

rec = JobRecommender()
rec.fit(df)
print('Fit OK, TF-IDF matrix built')

user_text = 'python sql pandas machine learning data analysis'
results = rec.recommend(user_text, top_k=5)
print('\nTop 5 matches:')
for i, row in results.iterrows():
    title = row['title'].title()[:40]
    co = row['company'].title()[:20]
    score = row['match_score']
    print(f'  {i+1}. {title} @ {co} | {score}%')

prios = priority_skills(results, ['python','sql'], top_n=5)
print('Priority skills:', [x['skill'] for x in prios])

assert len(results) == 5, 'Expected 5 results'
assert results['match_score'].max() > 0, 'No matches found'
print('\nEND-TO-END TEST PASSED')
