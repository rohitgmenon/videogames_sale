import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

REGIONS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
CATEGORIES = ["Platform", "Genre", "Publisher"]
FEATURES = CATEGORIES + ["Year"]

st.set_page_config(page_title="Video Game Sales", layout="wide")


@st.cache_data
def load_data():
    """Load and clean the data (same steps as the notebook)."""
    df = pd.read_csv("vgsales.csv")
    for col in ["Name", "Platform", "Genre", "Publisher"]:
        df[col] = df[col].str.strip()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.drop_duplicates(subset=["Name", "Platform", "Year"])
    df = df.dropna(subset=["Year"])
    df["Publisher"] = df["Publisher"].fillna("Unknown")
    df = df[df["Year"] <= 2016].copy()
    df["Year"] = df["Year"].astype(int)
    return df.drop(columns="Rank").reset_index(drop=True)


@st.cache_resource
def train_models(df):
    """Train an SVM hit classifier and a majority-class baseline."""
    X = df[FEATURES]
    y = (df["Global_Sales"] >= 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced",
            probability=True,
            random_state=42
        ),
    }

    fitted, rows = {}, []
    for name, model in models.items():
        encode = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20
                ), CATEGORIES),
                ("year", StandardScaler(), ["Year"]),
            ]
        )
        pipe = Pipeline([
            ("encode", encode),
            ("model", model),
        ]).fit(X_train, y_train)

        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, score),
        })

    results = pd.DataFrame(rows).set_index("Model").round(3)
    return fitted["SVM (RBF)"], "SVM (RBF)", results


def bar_chart(data, column, title):
    top = data.groupby(column)["Global_Sales"].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top.values, y=top.index.astype(str), color="steelblue", ax=ax)
    ax.set(title=title, xlabel="Global sales (millions)", ylabel="")
    st.pyplot(fig)
    plt.close(fig)

