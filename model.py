import joblib as jb
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer  

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline   

# =========================
# 1. Load data
# =========================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
#sub_df = pd.read_csv("sample_submission.csv")

TARGET = "Academic_Status"

# =========================
# 2. Fill NA cho TEXT
# =========================
for col in ["Advisor_Notes", "Personal_Essay"]:
    train_df[col] = train_df[col].fillna("")
    test_df[col] = test_df[col].fillna("")

# =========================
# 3. Feature groups
# =========================
base_numeric_features = [
    "Age",
    "Tuition_Debt",
    "Count_F",
    "Training_Score_Mixed"
]

attendance_features = [
    f"Att_Subject_{str(i).zfill(2)}" for i in range(1, 41)
]

numeric_features = base_numeric_features + attendance_features

categorical_features = [
    "Gender",
    "Hometown",
    "Current_Address",
    "Admission_Mode",
    "English_Level",
    "Club_Member"
]

# =========================
# 4. X, y
# =========================
X = train_df.drop(columns=[TARGET, "Student_ID"])
y = train_df[TARGET]

X_test = test_df.drop(columns=["Student_ID"])

print("X shape:", X.shape)
print("y shape:", y.shape)

# =========================
# 5. Preprocessor
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

text_transformer_advisor = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2)
    ))
])

text_transformer_essay = Pipeline(steps=[
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    ))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),

        # ✅ TRUYỀN STRING, KHÔNG PHẢI LIST
        ("advisor_text", text_transformer_advisor, "Advisor_Notes"),
        ("essay_text", text_transformer_essay, "Personal_Essay"),
    ],
    remainder="drop"
)

# =========================
# 6. FINAL MODEL PIPELINE (ĐÚNG)
# =========================
clf = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

# =========================
# 7. Train FULL
# =========================
print("Training on FULL training set...")
clf.fit(X, y)
jb.dump(clf,"model.pkl")

# =========================
# 8. Predict test
# =========================
#print("Predicting test set...")
#test_preds = clf.predict(X_test)


# =========================
# 9. Submission
# =========================
#sub_df["Academic_Status"] = test_preds
#sub_df.to_csv("submission.csv", index=False)

#print("\nSaved submission.csv")
#print(sub_df.head())
