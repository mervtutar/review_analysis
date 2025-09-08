# dashboard_table.py
# Tablo odaklı dashboard (sidebar yok) + community_likes + sadece is_fake
# Çalıştır: streamlit run dashboard_table.py

import os, re, textwrap
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

CSV_PATH = "patrol_officer_reviews_analyzed.csv"
COMM_RE = re.compile(r"community_liked\((\d+)\)", re.I)

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)

    # dtypes
    for c in ("score_stars","thumbs_up_count","cluster_size"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ("sentiment_score","interesting_score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)
    if "at" in df.columns:
        df["at"] = pd.to_datetime(df["at"], errors="coerce")
    for c in ("content","reply","sentiment_label","interesting_reason","user_name"):
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    for c in ("fake_flag","fake_flag_v2","interesting_flag"):
        if c in df.columns:
            df[c] = df[c].astype(bool)

    # community_likes: interesting_reason içindeki community_liked(XX) sayısı
    def extract_comm(x: str) -> int:
        m = COMM_RE.search(x or "")
        return int(m.group(1)) if m else 0
    df["community_likes"] = (
        df["interesting_reason"].apply(extract_comm).astype(int)
        if "interesting_reason" in df.columns else 0
    )

    # yalnızca is_fake (v2 tercih)
    if "fake_flag_v2" in df.columns:
        df["is_fake"] = df["fake_flag_v2"].astype(bool)
    elif "fake_flag" in df.columns:
        df["is_fake"] = df["fake_flag"].astype(bool)
    else:
        df["is_fake"] = False

    return df

def star_bar(df: pd.DataFrame):
    if "score_stars" not in df.columns or df.empty:
        st.info("Yıldız verisi yok.")
        return
    counts = df["score_stars"].value_counts().sort_index().reset_index()
    counts.columns = ["stars","count"]
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X("stars:O", title="Stars"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["stars","count"]
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

def sentiment_bar(df: pd.DataFrame):
    if "sentiment_label" not in df.columns or df.empty:
        st.info("Sentiment verisi yok.")
        return
    s = df["sentiment_label"].fillna("").astype(str).value_counts().reset_index()
    s.columns = ["sentiment","count"]
    chart = alt.Chart(s).mark_bar().encode(
        x=alt.X("sentiment:N", title="Sentiment"),
        y=alt.Y("count:Q", title="Count"),
        tooltip=["sentiment","count"]
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

# -------------------- App --------------------
st.set_page_config(page_title="Review Dashboard — Tablo", layout="wide")
st.title("🎮 Review Dashboard — Tablo")

try:
    df = load_csv(CSV_PATH)
except Exception as e:
    st.error(str(e))
    st.stop()

# KPI'lar
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Toplam yorum", f"{len(df):,}")
if "score_stars" in df.columns and len(df)>0:
    c2.metric("Ortalama ⭐", f"{df['score_stars'].mean():.2f}")
c3.metric("Fake %", f"{df['is_fake'].mean()*100:.1f}%")
if "interesting_flag" in df.columns and len(df)>0:
    c4.metric("Interesting %", f"{df['interesting_flag'].mean()*100:.1f}%")

st.divider()

# Grafikler
colA, colB = st.columns(2)
with colA:
    st.subheader("Yıldız Dağılımı")
    star_bar(df)
with colB:
    st.subheader("Sentiment Dağılımı")
    sentiment_bar(df)

st.divider()


st.subheader("Yorumlar (Tablo)")
show_cols = [c for c in [
    "at","score_stars","community_likes",
    "sentiment_label","sentiment_score",
    "is_fake",
    "interesting_flag","interesting_score","interesting_reason",
    "content","reply"
] if c in df.columns]

# Uzun metinleri kısalt
df_view = df.copy()
if "content" in df_view.columns:
    df_view["content"] = df_view["content"].apply(lambda t: textwrap.shorten(t or "", width=240, placeholder="…"))
if "reply" in df_view.columns:
    df_view["reply"] = df_view["reply"].apply(lambda t: textwrap.shorten(t or "", width=160, placeholder="…"))

table = df_view[show_cols].sort_values(
    by=[c for c in ["community_likes","interesting_score","at"] if c in df_view.columns],
    ascending=[False, False, False][:len([c for c in ["community_likes","interesting_score","at"] if c in df_view.columns])]
)

# Büyük veri için ilk 1000 satır
if len(table) > 1000:
    st.caption(f"Toplam {len(table):,} kayıt var; performans için ilk 1000 gösteriliyor.")
    table = table.head(1000)

st.dataframe(table, use_container_width=True, height=560)
