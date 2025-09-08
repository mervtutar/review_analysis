# dashboard.py
import streamlit as st
import pandas as pd

df = pd.read_csv("patrol_officer_reviews_analyzed.csv")

st.title("Patrol Officer Review Analysis")

st.metric("Toplam yorum", len(df))
st.metric("Fake %", f"{df['fake_flag'].mean()*100:.1f}%")
st.metric("Interesting %", f"{df['interesting_flag'].mean()*100:.1f}%")

st.subheader("Sentiment Dağılımı")
st.bar_chart(df["sentiment_label"].value_counts())

st.subheader("İlginç Yorumlar")
for _, row in df[df["interesting_flag"]].head(20).iterrows():
    st.write(f"⭐ {row['score_stars']} | 👍 {row['thumbs_up_count']} | {row['interesting_reason']}")
    st.write(row['content'])
    st.divider()
