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


st.title("🎮 Video Game Sales Explorer")

try:
    df = load_data()
except FileNotFoundError:
    st.error("vgsales.csv not found. Put it in the same folder as app.py.")
    st.stop()

data_tab, chart_tab, predict_tab = st.tabs(["Data", "Charts", "Predict a hit"])

with data_tab:
    c1, c2, c3 = st.columns(3)
    c1.metric("Games (cleaned rows)", f"{len(df):,}")
    c2.metric("Platforms", df["Platform"].nunique())
    c3.metric("Publishers", df["Publisher"].nunique())
    search = st.text_input("Search for a game")
    shown = df[df["Name"].str.contains(search, case=False, regex=False)] if search else df
    st.dataframe(shown, hide_index=True)

with chart_tab:
    genre = st.selectbox("Filter by genre", ["All"] + sorted(df["Genre"].unique()))
    view = df if genre == "All" else df[df["Genre"] == genre]
    left, right = st.columns(2)
    with left:
        bar_chart(view, "Name", "Top games")
        bar_chart(view, "Platform", "Top platforms")
    with right:
        bar_chart(view, "Publisher", "Top publishers")
        st.subheader("Sales by release year and region (millions)")
        yearly = view.groupby("Year")[REGIONS].sum()
        yearly.index = yearly.index.astype(str)
        st.line_chart(yearly)

with predict_tab:
    st.write("A **hit** is a game with at least 1 million global sales.")
    st.info(
        "This SVM estimates whether a game could become a hit from its planned "
        "platform, genre, publisher, and release year. It does not use the game's "
        "sales as an input."
    )
    st.warning(
        "The training data ends in 2016. Predictions for future years are "
        "extrapolations from older games and may be unreliable, especially for "
        "new platforms or publishers."
    )

    model, model_name, results = train_models(df)
    next_year = 2027
    with st.form("predict"):
        platform = st.text_input(
            "Platform (use a known value or enter a planned/new platform)",
            value=sorted(df["Platform"].unique())[0]
        ).strip()
        game_genre = st.selectbox("Genre", sorted(df["Genre"].dropna().unique()))
        publisher = st.text_input(
            "Publisher (use a known value or enter a planned/new publisher)",
            value="Unknown"
        ).strip() or "Unknown"
        year = st.number_input(
            "Planned release year",
            min_value=2017,
            max_value=2040,
            value=next_year,
            step=1
        )
        submitted = st.form_submit_button("Estimate hit potential")

    if submitted:
        game = pd.DataFrame(
            [[platform or "Unknown", game_genre, publisher, int(year)]],
            columns=FEATURES
        )
        score = model.predict_proba(game)[0, 1]
        predicted_class = int(score >= 0.5)
        if predicted_class:
            st.success(f"Estimated hit potential: HIGH (SVM score {score:.2f})")
        else:
            st.warning(f"Estimated hit potential: LOW (SVM score {score:.2f})")
        st.caption(
            "The score is the SVM's probability estimate, not a calibrated guarantee "
            "that the game will sell 1 million copies. Validate it on newer data "
            "before relying on future-year predictions."
        )

    st.subheader("SVM vs majority-class baseline")
    st.dataframe(results)
    st.caption(
        "Compare F1, Recall, and ROC-AUC as well as Accuracy. The baseline always "
        "predicts the most common class; a high baseline accuracy can hide poor "
        "hit detection."
    )
