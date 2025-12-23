import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report, confusion_matrix

# --- LOAD DATA ---
print("Loading data...")
try:
    df = pd.read_csv('global_exam_data.csv')
except FileNotFoundError:
    print("❌ Error: 'global_exam_data.csv' not found. Run data_generator.py first!")
    exit()

# --- PROJECT A: A/B TEST ---
print("\n--- PROJECT A: A/B TEST (Statistical Inference) ---")
group_mock = df[df['Used_Mock'] == 1]['Score']
group_no_mock = df[df['Used_Mock'] == 0]['Score']
t_stat, p_val = stats.ttest_ind(group_mock, group_no_mock)

print(f"Average Score with Mock: {group_mock.mean():.1f}")
print(f"Average Score without Mock: {group_no_mock.mean():.1f}")
print(f"P-Value: {p_val:.5f}")
if p_val < 0.05:
    print("✅ RESULT: Statistically significant difference found.")

# --- PREPROCESSING FOR ML ---
df_ml = pd.get_dummies(df, columns=['Location', 'Exam_Subject'], drop_first=True)
X = df_ml.drop(['Student_ID', 'Score', 'Result'], axis=1)
y_score = df_ml['Score']
y_pass = df['Result'].apply(lambda x: 1 if x == 'Pass' else 0)

# --- PROJECT B: LINEAR REGRESSION ---
print("\n--- PROJECT B: LINEAR REGRESSION (Score Prediction) ---")
X_train, X_test, y_train, y_test = train_test_split(X, y_score, test_size=0.2, random_state=42)
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
print(f"Model R-Squared: {r2_score(y_test, reg_model.predict(X_test)):.2%}")

# --- PROJECT C: CLASSIFICATION ---
print("\n--- PROJECT C: CLASSIFICATION (Pass/Fail Prediction) ---")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_pass, test_size=0.2, random_state=42)
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_c, y_train_c)
print(f"Accuracy: {accuracy_score(y_test_c, clf_model.predict(X_test_c)):.1%}")
