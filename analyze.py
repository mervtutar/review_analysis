# analyze_reviews_case2.py
# Case Study 2 — Sentiment + Fake v2 + Interesting
# Çalıştır: python analyze_reviews_case2.py
# Girdi : patrol_officer_reviews.csv
# Çıktı : patrol_officer_reviews_analyzed2.csv, patrol_officer_reviews_analyzed2.jsonl

import os, re, json, unicodedata, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

INPUT_CSV  = "patrol_officer_reviews.csv"
OUT_CSV    = "patrol_officer_reviews_analyzed.csv"
OUT_JSONL  = "patrol_officer_reviews_analyzed.jsonl"

# -------------------- Helpers --------------------
def normalize_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("I", "ı").replace("İ", "i")
    return unicodedata.normalize("NFKC", s).lower()

URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.I)
CONTACT_RE = re.compile(r"(@\w+)|(\+\d{6,})|([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
REPEAT_RE  = re.compile(r"(.)\1{4,}")

# -------------------- Sentiment --------------------
class SentimentEngine:
    """
    Primary: cardiffnlp/twitter-xlm-roberta-base-sentiment (çok dilli)
    Fallback: basit TR keyword kontrolü
    """
    def __init__(self, device: int = -1):
        self.pipe = None
        try:
            from transformers import pipeline
            self.pipe = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                device=device
            )
        except Exception:
            self.pipe = None

    def run(self, texts):
        if self.pipe is None:
            labels, scores = [], []
            POS = ("harika","mükemmel","çok iyi","super","süper","beğendim","efsane","muhteşem","bayıldım")
            NEG = ("kötü","rezalet","berbat","nefret ettim","iğrenç","hatalı","çöküyor","donuyor","bozuk")
            for t in texts:
                tl = (t or "").lower()
                if any(w in tl for w in POS): labels.append("positive"); scores.append(0.8)
                elif any(w in tl for w in NEG): labels.append("negative"); scores.append(0.8)
                else: labels.append("neutral"); scores.append(0.5)
            return labels, scores

        out = self.pipe(texts, truncation=True, batch_size=32)
        labels, scores = [], []
        for r in out:
            lab = r["label"].lower()
            if lab.startswith("neg"): lab="negative"
            elif lab.startswith("pos"): lab="positive"
            else: lab="neutral"
            labels.append(lab); scores.append(float(r.get("score",0.0)))
        return labels, scores

# -------------------- Fake Detection (v1 sinyalleri) --------------------
def fake_signals_v1(df):
    text = df["content"].astype(str).fillna("")
    norm = text.map(normalize_text)

    dup_exact = norm.duplicated(keep=False)

    def low_info_fn(s: str) -> bool:
        s = s or ""
        if len(s) < 6: return True
        if REPEAT_RE.search(s): return True
        letters = sum(ch.isalnum() for ch in s)
        return letters < 0.4 * max(1, len(s))
    low_info  = text.apply(low_info_fn)

    bot_like  = text.apply(lambda s: bool(URL_RE.search(s) or CONTACT_RE.search(s)))

    dup_near = [False]*len(df)
    cluster_labels = np.full(len(df), -1, dtype=int)

    # Near-dup (embeddings) + cluster
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.cluster import DBSCAN
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        E = model.encode(text.tolist(), convert_to_numpy=True, normalize_embeddings=True)
        sim = np.clip(E @ E.T, -1, 1)
        dist = 1 - sim
        clustering = DBSCAN(eps=0.22, min_samples=3, metric="precomputed").fit(dist)
        cluster_labels = clustering.labels_
        counts = pd.Series(cluster_labels[cluster_labels!=-1]).value_counts()
        suspicious = set(counts[counts>=5].index)  # uzun metinler için 5+
        dup_near = [int(l) in suspicious for l in cluster_labels]
    except Exception:
        pass

    fake_flag = dup_exact | low_info | bot_like | pd.Series(dup_near, index=df.index)
    reasons = []
    for i in range(len(df)):
        r = []
        if dup_exact.iloc[i]: r.append("dup_exact")
        if low_info.iloc[i]:  r.append("low_info")
        if bot_like.iloc[i]:  r.append("bot_like")
        if dup_near[i]:       r.append("dup_near")
        reasons.append(",".join(r))

    return fake_flag, reasons, dup_exact, pd.Series(dup_near, index=df.index), low_info, bot_like, cluster_labels

# -------------------- Fake v2 (organik şikâyet istisnalı) --------------------
def compute_fake_v2(df, dup_exact, dup_near, low_info, bot_like, cluster_labels,
                    short_len=20, common_short_min=30, burst_window_min=10,
                    burst_min_repeats=5, cluster_short_min=100, cluster_long_min=5):
    """
    - Kısa & yaygın övgüler masum (is_common_short)
    - Low-info tek başına yetmez; destek sinyali ister
    - Near-dup: kısa metinde cluster>=100, uzun metinde >=5 (organik şikâyette gevşet)
    - Burst/self-dup/URL güçlendirir
    - Organik reklam/süre/oyun şikâyetlerinde near-dup tek başına fake değildir
    """
    import re
    n = len(df)
    out = pd.DataFrame(index=df.index)
    txt = df["content"].astype(str).fillna("")
    out["len_chars"] = txt.str.len().astype(int)
    out["norm_txt"]  = txt.str.lower().str.replace(r"\s+"," ", regex=True).str.strip()

    # 0) Organik şikâyet maskesi
    complaint_re = re.compile(r"(reklam(lar)?|çok\s*reklam|ads?)|(\b\d+\s*(sn|saniye|dk|dakika)\b)", re.I)
    organic_complaint = txt.str.lower().apply(lambda s: bool(complaint_re.search(s))) & \
                        txt.str.lower().str.contains(r"(oyun|game)")

    # 1) common short whitelist
    short = out.loc[out["len_chars"]<=short_len, "norm_txt"]
    freq = short.value_counts()
    common_short = set(freq[freq>=common_short_min].index)
    out["is_common_short"] = out["norm_txt"].isin(common_short)
    exact_suspicious = dup_exact & ~out["is_common_short"]

    # 2) burst (10 dk içinde ≥5 tekrar)
    burst = pd.Series(False, index=df.index)
    if "at" in df.columns:
        times = pd.to_datetime(df["at"], errors="coerce")
        tmp = pd.DataFrame({"txt": out["norm_txt"], "at": times})
        tmp = tmp.dropna(subset=["at"]).sort_values("at")
        for txtv, grp in tmp.groupby("txt"):
            idxs = grp.index.tolist()
            ts = grp["at"].values
            if len(ts) < burst_min_repeats: continue
            L = 0
            for R in range(len(ts)):
                while ts[R] - ts[L] > np.timedelta64(burst_window_min, "m"):
                    L += 1
                if R - L + 1 >= burst_min_repeats:
                    burst.iloc[idxs[L:R+1]] = True
    out["burst_dup"] = burst

    # 3) cluster size
    cl_counts = pd.Series(cluster_labels).value_counts()
    out["cluster_size"] = pd.Series(cluster_labels).map(cl_counts).fillna(0).astype(int)

    # 4) near-dup eşikleri (organik şikâyette gevşet + tek başına kapat)
    short_mask = out["len_chars"] <= (short_len + 5)
    cluster_long_min_adj = np.where(organic_complaint, max(15, cluster_long_min), cluster_long_min)
    near_suspicious = (short_mask & (out["cluster_size"]>=cluster_short_min)) | \
                      (~short_mask & (out["cluster_size"]>=cluster_long_min_adj))
    # organik şikâyette near-dup tek başına fake sebebi olmasın:
    near_suspicious = near_suspicious & ~organic_complaint

    # 5) low-info destek ister
    low_info_support = low_info & (near_suspicious | bot_like | out["burst_dup"] | exact_suspicious)

    # 6) self-dup (aynı kullanıcı aynı metin)
    if "user_name" in df.columns:
        key = pd.MultiIndex.from_arrays([df["user_name"].fillna("").astype(str), out["norm_txt"]])
        self_dup = pd.Series(key.duplicated(keep=False), index=df.index)
    else:
        self_dup = pd.Series(False, index=df.index)

    # 7) final
    fake_v2 = (
        bot_like |
        near_suspicious |
        low_info_support |
        (exact_suspicious & ((out["cluster_size"]>=50) | out["burst_dup"] | self_dup))
    )

    # reason_v2 (CSV'ye bilgi amaçlı; dashboard'da göstermiyoruz)
    reasons_v2 = []
    for i in range(n):
        r = []
        if bool(bot_like.iloc[i]): r.append("bot_like")
        if bool(near_suspicious.iloc[i]): r.append(f"near_dup(cluster={int(out['cluster_size'].iloc[i])})")
        if bool(low_info_support.iloc[i]): r.append("low_info_support")
        if bool(exact_suspicious.iloc[i]): r.append("exact_dup_suspicious")
        if bool(out["burst_dup"].iloc[i]): r.append("burst_dup")
        if bool(self_dup.iloc[i]): r.append("self_dup")
        if bool(out["is_common_short"].iloc[i]): r.append("common_short_whitelist")
        if bool(organic_complaint.iloc[i]): r.append("organic_complaint")
        reasons_v2.append(",".join(r))

    return fake_v2, reasons_v2, out[["is_common_short","burst_dup","cluster_size"]]

# -------------------- Interesting --------------------
class InterestingEngine:
    """
    Zero-shot + rarity + sinyaller (uzunluk, emoji, ünlem, like, sentiment intensity).
    'community_liked' etiketi like sayısını içerir: community_liked(42)
    """
    def __init__(self, device=-1):
        self.pipe = None
        self.labels = ["humorous","constructive","exaggerated","suggestive","novel"]
        try:
            from transformers import pipeline
            self.pipe = pipeline("zero-shot-classification",
                                 model="joeddav/xlm-roberta-large-xnli",
                                 device=device)
        except Exception:
            self.pipe = None

    def rarity(self, texts):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1,2), lowercase=True)
        X = vect.fit_transform(texts)
        mean = np.asarray(X.mean(axis=1)).ravel()
        maxv = X.max(axis=1).toarray().ravel()
        return mean, maxv

    def run(self, df):
        txt   = df["content"].astype(str).fillna("")
        likes = df.get("thumbs_up_count", pd.Series([0]*len(df))).fillna(0).astype(int)

        L      = txt.str.len().values
        exclam = txt.apply(lambda s: s.count("!")+s.count("?")).values
        emojis = txt.apply(lambda s: sum(ch in s for ch in "😂🤣😅😁")).values
        sent_i = np.abs(df["sentiment_score"].fillna(0).astype(float).values)

        # rarity
        try:
            _, r_max = self.rarity(txt.tolist())
        except Exception:
            r_max = np.zeros(len(df))

        # zero-shot
        zs_scores = np.zeros((len(df), len(self.labels)))
        if self.pipe:
            out = self.pipe(txt.tolist(), candidate_labels=self.labels, multi_label=True, batch_size=16)
            for i, r in enumerate(out):
                mp = {lab.lower(): sc for lab, sc in zip(r["labels"], r["scores"])}
                zs_scores[i] = [mp.get(lbl, 0.0) for lbl in self.labels]

        # skor
        z = (0.20*np.tanh(L/220) + 0.15*np.tanh(exclam/3) + 0.15*np.tanh(likes/10) +
             0.10*np.tanh(emojis/1) + 0.15*np.tanh(sent_i/0.6) + 0.10*np.tanh(r_max/0.35) +
             0.15*np.tanh(zs_scores.sum(axis=1)))
        thr = np.quantile(z, 0.70)
        flags = z >= thr

        # reason —> community_liked(X)
        reasons = []
        for i in range(len(df)):
            r = []
            if L[i] >= 220: r.append("long")
            if exclam[i] >= 3: r.append("expressive")
            if likes.iloc[i] >= 10: r.append(f"community_liked({int(likes.iloc[i])})")
            if emojis[i] > 0: r.append("humorous")
            if sent_i[i] > 0.6: r.append("strong_sentiment")
            if r_max[i] > 0.35: r.append("rare_language")
            if self.pipe is not None:
                for k, lbl in enumerate(self.labels):
                    if zs_scores[i, k] >= 0.5:
                        r.append(f"zs_{lbl}")
            if not r:
                r.append("novelty")
            reasons.append(",".join(r))

        return flags, z, reasons

# -------------------- Main --------------------
def main():
    assert os.path.isfile(INPUT_CSV), f"Girdi bulunamadı: {INPUT_CSV}"
    df = pd.read_csv(INPUT_CSV)

    # Temizlik
    for c in ["content","reply","user_name"]:
        if c in df.columns: df[c] = df[c].fillna("").astype(str)
    if "thumbs_up_count" in df.columns:
        df["thumbs_up_count"] = pd.to_numeric(df["thumbs_up_count"], errors="coerce").fillna(0).astype(int)
    if "score_stars" in df.columns:
        df["score_stars"] = pd.to_numeric(df["score_stars"], errors="coerce").fillna(0).astype(int)

    # Sentiment
    print("▶ Sentiment")
    sent = SentimentEngine()
    df["sentiment_label"], df["sentiment_score"] = sent.run(df["content"].tolist())

    # Fake v1 sinyalleri
    print("▶ Fake (sinyaller)")
    fake_flag_v1, fake_reason_v1, dup_exact, dup_near, low_info, bot_like, cluster_labels = fake_signals_v1(df)
    df["dup_exact"]   = dup_exact.values if hasattr(dup_exact, "values") else dup_exact
    df["dup_near"]    = dup_near.values if hasattr(dup_near, "values") else dup_near
    df["low_info"]    = low_info.values if hasattr(low_info, "values") else low_info
    df["bot_like"]    = bot_like.values if hasattr(bot_like, "values") else bot_like
    df["cluster_id"]  = cluster_labels
    df["fake_flag"]   = fake_flag_v1
    df["fake_reason"] = fake_reason_v1

    # Fake v2 (organik şikâyet istisnalı)
    print("▶ Fake v2 (iyileştirilmiş)")
    fake_v2, reason_v2, aux = compute_fake_v2(
        df, dup_exact, dup_near, low_info, bot_like, cluster_labels,
        short_len=20, common_short_min=30, burst_window_min=10,
        burst_min_repeats=5, cluster_short_min=100, cluster_long_min=5
    )
    df["fake_flag_v2"]     = fake_v2.values if hasattr(fake_v2, "values") else fake_v2
    df["fake_reason_v2"]   = reason_v2
    df["is_common_short"]  = aux["is_common_short"].values
    df["burst_dup"]        = aux["burst_dup"].values
    df["cluster_size"]     = aux["cluster_size"].values

    # Interesting
    print("▶ Interesting")
    intr = InterestingEngine()
    fl, sc, rs = intr.run(df)
    df["interesting_flag"]   = fl
    df["interesting_score"]  = sc
    df["interesting_reason"] = rs

    # Kaydet
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    # Özet
    summary = {
        "n": int(len(df)),
        "fake_v2_%": float(pd.Series(df["fake_flag_v2"]).mean()*100),
        "interesting_%": float(pd.Series(df["interesting_flag"]).mean()*100),
    }
    print("✅ Bitti\n", json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"CSV → {OUT_CSV}\nJSONL → {OUT_JSONL}")

if __name__ == "__main__":
    main()
