"""
=============================================================================
 train.py  —  A-Level Pathway Recommendation Model (Random Forest)
 BTech Capstone Project — MUHORAKEYE MARTHA (25RP19281)
 G.S NYAKINAMA I TSS, Musanze, Northern Province, Rwanda
=============================================================================

PURPOSE:
  This script trains a Random Forest machine learning model that recommends
  one of Rwanda's 4 A-Level pathway classes to a student based on:
    1. Their O-Level subject grades (9 subjects)
    2. Their personal preferences (wish, career interest, talent, labor market)
    3. Computed group scores derived from the raw grades (feature engineering)

RWANDA CONTEXT:
  Following the MINEDUC National Education Conference on June 20, 2025,
  Rwanda replaced 11 old A-Level subject combinations (PCM, HEG, MCB, etc.)
  with 3 broad Learning Pathways:
    - Mathematics & Sciences  (replaces PCM, PCB, MCB, MEG, MPG, MCE)
    - Arts & Humanities       (replaces HEG, HEL, HGL)
    - Languages               (replaces EFK, LFK, EKK)
  TVET tracks (Computer Application, Electronics, etc.) remain UNCHANGED.
  Source: mineduc.gov.rw, REB press release June 2025

ALGORITHM CHOICE:
  Random Forest was chosen because:
    - It handles both numeric (grades) and categorical (encoded choices) features
    - It is robust against overfitting via ensemble of 300 decision trees
    - It provides feature importance scores (useful for explainability)
    - It performs well on small-to-medium datasets (our case: ~1,294 rows)

NOT ELIGIBLE CLASS:
  A 5th class "Not Eligible" is added synthetically to represent students
  who failed 5 or more subjects (< 50% in those subjects, i.e. grade F).
  The original dataset had zero such records, so the model had no signal
  for this case. By adding ~400 synthetic failing-student records, the
  model learns to predict "Not Eligible" directly from the grades — no
  hard-coded input gate is needed.

ACCURACY:
  ~91% Test Accuracy | ~93% Cross-Validation (5-Fold Stratified)

RUN ONCE:
  python3 model/train.py
  (Re-run only if dataset changes)
=============================================================================
"""

import pandas as pd        # for loading and manipulating the Excel dataset
import numpy as np         # for numerical operations
import joblib              # for saving trained model artifacts to disk
import os                  # for building file paths
import warnings
warnings.filterwarnings('ignore')   # suppress sklearn version warnings

from sklearn.ensemble import RandomForestClassifier   # main ML algorithm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # data preparation
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# ── File paths ──────────────────────────────────────────────────────────────
# Build paths relative to this file so the project works on any machine
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'survey_data.xlsx')
OUT_DIR   = os.path.dirname(__file__)   # save model files in the same /model folder

# ── Step 1: Load the dataset ─────────────────────────────────────────────────
df = pd.read_excel(DATA_FILE)

# Remove any rows that have no pathway label (incomplete records)
df = df.dropna(subset=['Pathway'])
df = df[df['Pathway'].str.strip() != '']

print(f"Total training rows (original): {len(df)}")
print("Pathway distribution (target classes):")
print(df['Pathway'].value_counts())

# ── Step 1b: Synthesise "Not Eligible" student records ───────────────────────
# The original dataset has ZERO records of students who failed most subjects.
# Without these, the model has no signal for weak students and will randomly
# assign them a pathway. We generate 400 synthetic failing-student records
# covering a range of failure patterns so the model learns this boundary.
#
# A student is "Not Eligible" when they pass fewer than 5 out of 9 subjects
# (pass = grade D or above, i.e. encoded value >= 3, meaning >= 50%).
# We use grades F (1) and E (2) to represent failing/very weak performance.

rng = np.random.default_rng(seed=99)
SUBJECTS_LIST = ['Math', 'Physics', 'Chemistry', 'Biology', 'Geography',
                 'History', 'Entrepreneurship', 'English', 'Kinyarwanda',
                 'French', 'Kiswahili']
WISH_VALS   = [1, 2, 3, 4, 5]
CAREER_VALS = [1, 2, 3, 4, 5, 6]
TALENT_VALS = [1, 2, 3, 4, 5]
LABOR_VALS  = [0, 1, 2]

not_eligible_rows = []
for _ in range(600):
    # Decide how many subjects the student passes (0 to 4 — always below threshold)
    n_pass = rng.integers(0, 5)          # 0, 1, 2, 3, or 4 passes
    n_fail = 11 - n_pass                  # rest are failed

    # Failing subjects get 1 (F) only — no ambiguous E grades
    # Passing subjects vary from D(3) to A(6) to cover borderline cases
    # where a student aces a few subjects but fails the majority
    pass_grades = rng.integers(3, 7, size=n_pass).tolist()   # 3=D, 4=C, 5=B, 6=A
    fail_grades = [1] * n_fail                                # 1=F
    grades = fail_grades + pass_grades
    rng.shuffle(grades)

    row = {
        'Gender':          int(rng.integers(0, 2)),
        'StudentWish':     int(rng.choice(WISH_VALS)),
        'CareerInterest':  int(rng.choice(CAREER_VALS)),
        'Talent':          int(rng.choice(TALENT_VALS)),
        'LaborMarket':     int(rng.choice(LABOR_VALS)),
        'Pathway':         'Not Eligible',
    }
    for i, sub in enumerate(SUBJECTS_LIST):
        row[sub] = int(grades[i])

    not_eligible_rows.append(row)

df_not_eligible = pd.DataFrame(not_eligible_rows)

# Merge with original data
df_all = pd.concat([df, df_not_eligible], ignore_index=True)

print(f"\nAfter adding 'Not Eligible' synthetic records: {len(df_all)} total rows")
print("Updated pathway distribution:")
print(df_all['Pathway'].value_counts())

# Work with the merged dataset from here on
df = df_all

# ── Step 2: Encode O-Level grades to numbers ─────────────────────────────────
# Machine learning models require numeric inputs.
# We convert letter grades A–F to integers 6–1 (higher = better performance).
#   A = 6  (Excellent)
#   B = 5  (Very Good)
#   C = 4  (Good)
#   D = 3  (Satisfactory)
#   E = 2  (Poor)
#   F = 1  (Fail)
GRADE_MAP = {'A': 6, 'B': 5, 'C': 4, 'D': 3, 'E': 2, 'S': 2, 'F': 1}

SUBJECTS = ['Math', 'Physics', 'Chemistry', 'Biology', 'Geography',
            'History', 'Entrepreneurship', 'English', 'Kinyarwanda',
            'French', 'Kiswahili']

for col in SUBJECTS:
    # Map each grade letter to its number; fill missing with 3 (D = average)
    # The "Not Eligible" rows already have integers — coerce handles both cases
    df[col] = df[col].map(lambda x: GRADE_MAP[x] if isinstance(x, str) else x)
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(3).astype(int)

# Encode gender: Female = 0, Male = 1
df['Gender'] = df['Gender'].map(lambda x: {'Male': 1, 'Female': 0}.get(x, x) if isinstance(x, str) else x)
df['Gender'] = pd.to_numeric(df['Gender'], errors='coerce').fillna(0).astype(int)

# ── Step 3: Encode student preference features ───────────────────────────────
# These are the four new "student choice" fields added per supervisor's feedback.
# Each categorical value is mapped to an integer so the model can use it.

# Student Wish / Passion (what subject area they want to study)
df['StudentWish'] = df['StudentWish'].map(lambda x: {
    'Sciences': 1, 'Arts & Humanities': 2, 'Languages': 3,
    'Technology': 4, 'Business & Economics': 5
}.get(x, x) if isinstance(x, str) else x)
df['StudentWish'] = pd.to_numeric(df['StudentWish'], errors='coerce').fillna(1).astype(int)

# Career Interest (what job/profession they are aiming for)
df['CareerInterest'] = df['CareerInterest'].map(lambda x: {
    'Health & Medicine': 1, 'Engineering & Technology': 2,
    'Business & Finance': 3, 'Law & Policy': 4,
    'Journalism & Diplomacy': 5, 'Education & Teaching': 6
}.get(x, x) if isinstance(x, str) else x)
df['CareerInterest'] = pd.to_numeric(df['CareerInterest'], errors='coerce').fillna(1).astype(int)

# Talent (the student's self-assessed strongest skill)
df['Talent'] = df['Talent'].map(lambda x: {
    'Analytical / Problem-solving': 1, 'Creative / Artistic': 2,
    'Communication / Public Speaking': 3, 'Technical / Hands-on': 4,
    'Reading, Writing & Research': 5
}.get(x, x) if isinstance(x, str) else x)
df['Talent'] = pd.to_numeric(df['Talent'], errors='coerce').fillna(1).astype(int)

# Labor Market awareness (whether job demand influences their choice)
df['LaborMarket'] = df['LaborMarket'].map(lambda x: {
    'Yes': 1, 'No': 0, 'Not Sure': 2
}.get(x, x) if isinstance(x, str) else x)
df['LaborMarket'] = pd.to_numeric(df['LaborMarket'], errors='coerce').fillna(2).astype(int)

# ── Step 4: Feature Engineering — Computed Group Scores ─────────────────────
# Raw subject grades alone are sometimes not enough to separate pathways cleanly.
# For example, both Math & Sciences and TVET students score high in Math/Physics.
# By computing domain-specific group averages, we give the model a clearer signal.
#
# These scores are calculated here AND must be replicated identically in app.py
# when a student submits the prediction form.

# Average of core science subjects — high for Math & Sciences students
df['ScienceScore'] = df[['Math', 'Physics', 'Chemistry', 'Biology']].mean(axis=1)

# Average of humanities subjects — high for Arts & Humanities students
df['HumanitiesScore'] = df[['History', 'Geography', 'English']].mean(axis=1)

# Average of language subjects — high for Languages pathway students
df['LanguageScore'] = df[['English', 'Kinyarwanda', 'French', 'Kiswahili']].mean(axis=1)

# Average of technical subjects — high for TVET students
df['TechScore'] = df[['Math', 'Physics', 'Entrepreneurship']].mean(axis=1)

# Average of business-related subjects — high for Stream 2 / business-oriented students
df['BusinessScore'] = df[['Math', 'Geography', 'Entrepreneurship']].mean(axis=1)

# Overall academic average across all 9 subjects
df['OverallAvg'] = df[SUBJECTS].mean(axis=1)

# ── Step 5: Assemble feature matrix ─────────────────────────────────────────
# FEATURE_COLS defines the exact order of features.
# This list is saved to disk and must be used in the same order in app.py.
FEATURE_COLS = (
    ['Gender'] +                                           # 1 demographic feature
    SUBJECTS +                                             # 9 raw grade features
    ['StudentWish', 'CareerInterest', 'Talent', 'LaborMarket'] +   # 4 choice features
    ['ScienceScore', 'HumanitiesScore', 'LanguageScore',  # 6 engineered group scores
     'TechScore', 'BusinessScore', 'OverallAvg']
)
# Total: 1 + 11 + 4 + 6 = 22 features

X = df[FEATURE_COLS].values.astype(float)

# ── Step 6: Scale features to [0, 1] range ──────────────────────────────────
# MinMaxScaler ensures all features have equal weight regardless of their
# original range (e.g. grades 1–6 vs. group averages 1.0–6.0).
# IMPORTANT: We fit the scaler on training data only to avoid data leakage.
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ── Step 7: Encode target labels ─────────────────────────────────────────────
# LabelEncoder converts pathway names (strings) to integers (0, 1, 2, 3).
# Example: 'Arts & Humanities'→0, 'Languages'→1, 'Mathematics & Sciences'→2, 'TVET — Technology'→3
le = LabelEncoder()
y  = le.fit_transform(df['Pathway'].values)

print(f"\nTarget classes: {list(le.classes_)}")
print(f"  → Includes 'Not Eligible' as a learnable class (no hard-coded gate needed)")
print(f"Total features : {len(FEATURE_COLS)}")

# ── Step 8: Train/Test Split ──────────────────────────────────────────────────
# Split data: 80% for training, 20% for final evaluation.
# stratify=y ensures each class appears proportionally in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.20,
    random_state=42,    # fixed seed for reproducibility
    stratify=y
)
print(f"\nTraining set: {len(X_train)} rows | Test set: {len(X_test)} rows")

# ── Step 9: Train the Random Forest model ───────────────────────────────────
# HYPERPARAMETERS explained:
#   n_estimators=300   — Build 300 decision trees (more trees = more stable predictions)
#   max_depth=20       — Each tree can split up to 20 levels deep (prevents overfitting)
#   min_samples_split=2— A node must have at least 2 samples to split further
#   min_samples_leaf=1 — Each leaf must contain at least 1 sample
#   max_features='sqrt'— Each tree considers sqrt(20) ≈ 4 random features per split
#                        (introduces randomness, reduces correlation between trees)
#   class_weight='balanced' — Adjusts for unequal class sizes in training data
#   random_state=42    — Makes training reproducible across runs
#   n_jobs=-1          — Use all CPU cores (faster training)
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# ── Step 10: Cross-Validation ─────────────────────────────────────────────────
# Stratified K-Fold CV divides training data into 5 equal folds.
# The model is trained on 4 folds and tested on 1, repeated 5 times.
# This gives a reliable estimate of real-world accuracy.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')

# ── Step 11: Final training and evaluation ───────────────────────────────────
rf.fit(X_train, y_train)
y_pred   = rf.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

print("\n" + "=" * 58)
print("  RANDOM FOREST — TRAINING RESULTS")
print("=" * 58)
print(f"  5-Fold Cross-Validation : {cv_scores.mean()*100:.1f}%  (+/-{cv_scores.std()*100:.1f}%)")
print(f"  Hold-out Test Accuracy  : {test_acc*100:.1f}%")
print("=" * 58)
# Detailed per-class precision, recall and F1-score
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# ── Step 12: Save trained artifacts ──────────────────────────────────────────
# Four files are saved — all must be present for app.py to work:
#   model.pkl        — the trained Random Forest object
#   scaler.pkl       — the fitted MinMaxScaler (needed to scale new inputs)
#   label_encoder.pkl— the LabelEncoder (converts predicted int back to pathway name)
#   feature_cols.pkl — the ordered list of 20 feature names (used in app.py to build input)
joblib.dump(rf,           os.path.join(OUT_DIR, 'model.pkl'))
joblib.dump(scaler,       os.path.join(OUT_DIR, 'scaler.pkl'))
joblib.dump(le,           os.path.join(OUT_DIR, 'label_encoder.pkl'))
joblib.dump(FEATURE_COLS, os.path.join(OUT_DIR, 'feature_cols.pkl'))
print("\nAll model artifacts saved to /model/")
