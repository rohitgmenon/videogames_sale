import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

REGIONS = ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
CATEGORIES = ["Platform", "Genre", "Publisher"]
MODEL_CATEGORIES = ["Genre", "Publisher"]  # Platform is not a model input
FEATURES = MODEL_CATEGORIES + ["Year"]

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
def train_models(_df):
    """Train an SVM hit classifier and a majority-class baseline."""
    X = _df[FEATURES]
    y = (_df["Global_Sales"] >= 1).astype(int)
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
            random_state=42,
        ),
    }

    fitted, rows = {}, []
    for name, model in models.items():
        encode = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=20,
                    ),
                    MODEL_CATEGORIES,
                ),
                ("year", StandardScaler(), ["Year"]),
            ]
        )
        pipe = Pipeline([("encode", encode), ("model", model)]).fit(X_train, y_train)

        pred = pipe.predict(X_test)
        score = pipe.predict_proba(X_test)[:, 1]
        fitted[name] = pipe
        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, score),
            }
        )

    results = pd.DataFrame(rows).set_index("Model").round(3)
    return fitted["SVM (RBF)"], "SVM (RBF)", results


def bar_chart(data, column, title, value="Global_Sales"):
    top = data.groupby(column)[value].sum().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=top.values, y=top.index.astype(str), color="steelblue", ax=ax)
    ax.set(title=title, xlabel="Sales (millions)", ylabel="")
    st.pyplot(fig)
    plt.close(fig)


def trend_chart(data, value="Global_Sales"):
    yearly = data.groupby("Year")[value].sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(x=yearly.index, y=yearly.values, marker="o", color="steelblue", ax=ax)
    ax.set(title=f"{value} by release year", xlabel="Year", ylabel="Sales (millions)")
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- app body
st.title("Video Game Sales Explorer")

try:
    df = load_data()
except FileNotFoundError:
    st.error("vgsales.csv not found. Put it in the same folder as app.py and run "
             "`streamlit run app.py` from that folder.")
    st.stop()

with st.spinner("Training models (first run can take a minute)..."):
    model, model_name, results = train_models(df)

# Sidebar filters apply to the charts tab
st.sidebar.header("Filters")
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Release year", year_min, year_max, (year_min, year_max))
genres = st.sidebar.multiselect("Genre", sorted(df["Genre"].unique()))

filtered = df[df["Year"].between(*year_range)]
if genres:
    filtered = filtered[filtered["Genre"].isin(genres)]

tab1, tab2, tab3 = st.tabs(["Sales Charts", "Hit Predictor", "Data"])

with tab1:
    if filtered.empty:
        st.warning("No games match the current filters.")
    else:
        c1, c2 = st.columns(2)
        value = c1.selectbox("Sales region", ["Global_Sales"] + REGIONS)
        group = c2.selectbox("Group by", CATEGORIES)

        left, right = st.columns(2)
        with left:
            bar_chart(filtered, group, f"Top 10 {group} by {value}", value)
        with right:
            trend_chart(filtered, value)

with tab2:
    st.subheader(f"Model performance ({model_name} vs baseline)")
    st.caption("A game counts as a 'hit' if it sold 1 million copies or more globally.")
    st.dataframe(results, use_container_width=True)

    st.subheader("Predict a hit")
    c1, c2, c3 = st.columns(3)
    genre = c1.selectbox("Genre ", sorted(df["Genre"].unique()))
    publisher = c2.selectbox("Publisher", sorted(df["Publisher"].unique()))
    year = c3.slider("Release year", 2017, 2040, 2017)

    if st.button("Predict"):
        row = pd.DataFrame(
            [{"Genre": genre, "Publisher": publisher, "Year": year}]
        )
        prob = model.predict_proba(row)[0, 1]
        st.metric("Probability of selling 1M+ copies", f"{prob:.1%}")
        st.caption(
            f"The training data ends in {year_max}, so a {year} release is an "
            "extrapolation. Treat this as a rough guide, not a forecast."
        )

with tab3:
    st.write(f"{len(df):,} games after cleaning")
    st.dataframe(df, use_container_width=True)
    
