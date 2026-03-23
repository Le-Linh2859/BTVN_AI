import pandas as pd
import numpy as np
import re
import pickle

# ================= TOKENIZER =================
def simple_vn_tokenize(text):
    if pd.isna(text) or text == "":
        return ""
    text = str(text).lower()
    tokens = re.findall(r'[a-záàảãạăắặẳẵằâấầẩẫậđéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵ0-9]+', text)
    return " ".join(tokens)

# ================= FEATURE =================
def apply_feature_engineering(df):
    df = df.copy()
    
    for col in ["Advisor_Notes", "Personal_Essay"]:
        df[col] = df[col].fillna("")
        df[col] = df[col].apply(simple_vn_tokenize)

    return df

# ================= LOAD =================
def load_model():
    model = pickle.load(open("catboost_model.pkl", "rb"))
    tfidf = pickle.load(open("tfidf.pkl", "rb"))
    svd = pickle.load(open("svd.pkl", "rb"))
    return model, tfidf, svd

# ================= PREDICT =================
def predict(df):
    model, tfidf, svd = load_model()

    df = apply_feature_engineering(df)

    text = df["Advisor_Notes"] + " " + df["Personal_Essay"]
    X_text = svd.transform(tfidf.transform(text))

    X_final = np.hstack([df.select_dtypes(include=np.number).values, X_text])

    preds = model.predict(X_final)
    return preds