import pandas as pd
import numpy as np
import re
import pickle
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ================================================================
# 1. TOKENIZER
# ================================================================
def simple_vn_tokenize(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower()
    tokens = re.findall(r'[a-záàảãạăắặẳẵằâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9]+', text)
    return " ".join(tokens)

# ================================================================
# 2. FEATURE ENGINEERING
# ================================================================
def apply_feature_engineering(df):
    df = df.copy()
    
    # Text features
    for col in ["Advisor_Notes", "Personal_Essay"]:
        df[col] = df[col].fillna("")
        df[f"{col}_len"] = df[col].apply(len)
        df[f"{col}_words"] = df[col].apply(lambda x: len(x.split()))
        df[col] = df[col].apply(simple_vn_tokenize)

    # Attendance features
    attendance_cols = [f"Att_Subject_{str(i).zfill(2)}" for i in range(1, 41)]
    att_values = df[attendance_cols].values

    valid_att = np.where((att_values >= 0) & (att_values <= 20), att_values, np.nan)

    df["Att_Mean"] = np.nanmean(valid_att, axis=1)
    df["Att_Std"] = np.nanstd(valid_att, axis=1)
    df["Att_Min"] = np.nanmin(valid_att, axis=1)
    df["Att_Critical_Count"] = np.sum(valid_att < 5, axis=1)
    df["Att_Registered"] = np.sum(~np.isnan(valid_att), axis=1)

    # Cross features
    df["Is_Debtor"] = (df["Tuition_Debt"] > 0).astype(int)
    df["Debt_Age_Ratio"] = df["Tuition_Debt"] / (df["Age"] + 1)
    df["Total_Risk_Score"] = df["Count_F"] + df["Att_Critical_Count"]

    df = df.fillna(0)
    return df

# ================================================================
# 3. LOAD DATA
# ================================================================
print("Loading data...")
train_df = pd.read_csv("train.csv")

train_df = apply_feature_engineering(train_df)

TARGET = "Academic_Status"
REDUNDANT_COLS = [TARGET, "Student_ID", "Advisor_Notes", "Personal_Essay"]

X = train_df.drop(columns=REDUNDANT_COLS)
y = train_df[TARGET]

# categorical
cat_features = ["Gender", "Hometown", "Current_Address", "Admission_Mode", "English_Level", "Club_Member"]
for col in cat_features:
    X[col] = X[col].astype(str)

# ================================================================
# 4. TEXT PROCESSING
# ================================================================
print("Processing text...")

full_text = train_df["Advisor_Notes"] + " " + train_df["Personal_Essay"]

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, min_df=2)
tfidf.fit(full_text)

svd = TruncatedSVD(n_components=30, random_state=42)
svd.fit(tfidf.transform(full_text))

def transform_text(df):
    text = df["Advisor_Notes"] + " " + df["Personal_Essay"]
    return svd.transform(tfidf.transform(text))

X_text = transform_text(train_df)

# Combine features
X_final = np.hstack([X.values, X_text])

cat_indices = [list(X.columns).index(c) for c in cat_features]

# ================================================================
# 5. TRAIN MODEL (CV)
# ================================================================
print("Training model...")

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = []
scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_final, y)):
    print(f"Fold {fold+1}")

    X_train, X_val = X_final[train_idx], X_final[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        iterations=1500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5,
        loss_function='MultiClass',
        eval_metric='TotalF1',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=100,
        cat_features=cat_indices,
        class_weights=[1.0, 2.0, 3.5]
    )

    model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)

    preds = model.predict(X_val)
    f1 = f1_score(y_val, preds, average='macro')

    print(f"F1: {f1:.4f}")

    models.append(model)
    scores.append(f1)

print("Mean F1:", np.mean(scores))

# ================================================================
# 6. SAVE MODEL
# ================================================================
print("Saving model...")

# lấy model tốt nhất
best_model = models[np.argmax(scores)]

pickle.dump(best_model, open("catboost_model.pkl", "wb"))
pickle.dump(tfidf, open("tfidf.pkl", "wb"))
pickle.dump(svd, open("svd.pkl", "wb"))

print("DONE!")