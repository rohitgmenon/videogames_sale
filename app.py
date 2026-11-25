import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
    """Train the models and return the best one with a score table."""
    X = df[FEATURES]
    y = (df["Global_Sales"] >= 1).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                                class_weight="balanced", random_state=42),
    }
    fitted, rows = {}, []
    for name, model in models.items():
        encode = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), CATEGORIES)],
            remainder="passthrough")
        pipe = Pipeline([("encode", encode), ("model", model)]).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred, zero_division=0),
                     "Recall": recall_score(y_test, pred),
                     "F1": f1_score(y_test, pred),
                     "ROC-AUC": roc_auc_score(y_test, score)})
    results = pd.DataFrame(rows).set_index("Model").round(3)
    best_name = results.drop("Baseline")["ROC-AUC"].idxmax()
    return fitted[best_name], best_name, results


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
    model, model_name, results = train_models(df)
    with st.form("predict"):
        platform = st.selectbox("Platform", sorted(df["Platform"].unique()))
        game_genre = st.selectbox("Genre", sorted(df["Genre"].unique()))
        publisher = st.selectbox("Publisher", sorted(df["Publisher"].unique()))
        year = st.slider("Release year", int(df["Year"].min()), 2016, 2015)
        submitted = st.form_submit_button("Predict")
    if submitted:
        game = pd.DataFrame([[platform, game_genre, publisher, year]], columns=FEATURES)
        score = model.predict_proba(game)[0, 1]
        if score >= 0.5:
            st.success(f"Likely a HIT (hit score {score:.2f})")
        else:
            st.warning(f"Probably not a hit (hit score {score:.2f})")
        st.caption("The hit score is not an exact probability.")
    st.subheader(f"Model comparison (using: {model_name})")
    st.dataframe(results)
