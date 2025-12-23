import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(99)

# 1. SETUP PARAMETERS
num_rows = 1000
subjects = ['Financial Maths', 'Derivatives', 'Risk Management', 'Ethics', 'Corporate Finance']
countries = ['UK', 'USA', 'Singapore', 'Hong Kong', 'Germany', 'France']
country_weights = [0.6, 0.1, 0.1, 0.1, 0.05, 0.05]

# 2. GENERATE BASE COLUMNS
data = {
    'Student_ID': [f'STU{i:05d}' for i in range(num_rows)],
    'Exam_Subject': np.random.choice(subjects, num_rows),
    'Location': np.random.choice(countries, num_rows, p=country_weights),
    'Has_Degree': np.random.choice([0, 1], num_rows, p=[0.4, 0.6]), 
    'Used_Mock': np.random.choice([0, 1], num_rows, p=[0.5, 0.5]),  
    'Attempt_Count': np.random.choice([1, 2, 3, 4, 5, 6], num_rows, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
}

df = pd.DataFrame(data)

# 3. GENERATE SCORES
base_score = np.random.normal(55, 12, num_rows)
mock_boost = df['Used_Mock'] * np.random.normal(7, 2, num_rows)
degree_boost = df['Has_Degree'] * np.random.normal(3, 1, num_rows)
attempt_penalty = (df['Attempt_Count'] - 1) * 3

df['Score'] = base_score + mock_boost + degree_boost - attempt_penalty
df['Score'] = df['Score'].clip(0, 100).round(0)
df['Result'] = np.where(df['Score'] >= 50, 'Pass', 'Fail')

# 4. SAVE
df.to_csv('global_exam_data.csv', index=False)
print("✅ 'global_exam_data.csv' created.")
