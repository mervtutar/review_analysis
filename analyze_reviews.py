# analyze_reviews_model_based.py
# Case Study 2 — Review Analysis (Model Based, Fixed Input/Output)

import os, re, json, math, unicodedata, hashlib, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------- Sabit dosya adları --------------------
INPUT_CSV  = "patrol_officer_reviews.csv"
OUT_CSV    = "patrol_officer_reviews_analyzed.csv"
OUT_JSONL  = "patrol_officer_reviews_analyzed.jsonl"

# -------------------- Yardımcılar --------------------
def normalize_tr(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("I", "ı").replace("İ", "i")
    s = unicodedata.normalize("NFKC", s)
    return s.lower()

URL_RE     = re.compile(r"https?://\S+|www\.\S+", re.I)
CONTACT_RE = re.compile(r"(@\w+)|(\+\d{6,})|([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
REPEAT_RE  = re.compile(r"(.)\1{4,}")
TOKEN_RE   = re.compile(r"[a-zçğıöşü]+", re.I)

POS_TR = set("harika mükemmel çok iyi güzel şahane süper beğendim sevdim eğlenceli keyifli efsane muhteşem başarılı kaliteli bayıldım tavsiye ederim memnun tatmin".split())
NEG_TR = set("kötü berbat rezalet iğrenç saçma sinir bozucu donuyor çöküyor bug hata hatalı sorun sorunlu lag kasıyor yavaş reklam dolu gereksiz nefret ettim çalışmıyor açılmıyor bozuk hile hileli iade".split())

LAUGH_TOKENS        = ("😂","🤣","😅","😁","lol","lmao","haha","ahah","xd")
CONSTRUCTIVE_TOKENS = ("please","lütfen","öneri","suggest","düzeltilmeli","fix","improve","geliştirin","ekleyin","feature","özellik")
EXAG_TOKENS         = ("mükemmel","rezalet","asla","aşırı","inanılmaz","en iyi","en kötü","tam bir","kesinlikle","nefret")

def safe_series(df, name, default=0):
    return df[name] if name in df.columns else pd.Series([default]*len(df))

# -------------------- Sentiment --------------------
class SentimentEngine:
    def __init__(self, device: int = -1):
        self.pipe = None
        try:
            from transformers import pipeline
            model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
            self.pipe = pipeline("sentiment-analysis", model=model_id, device=device)
        except Exception:
            self.pipe = None

    def _rule_score(self, text: str) -> float:
        t = normalize_tr(text)
        toks = TOKEN_RE.findall(t)
        if not toks: return 0.0
        pos = sum(1 for w in toks if w in POS_TR)
        neg = sum(1 for w in toks if w in NEG_TR)
        return (pos - neg) / max(5, len(toks))

    def _label_from_score(self, x: float) -> str:
        if x > 0.03: return "positive"
        if x < -0.03: return "negative"
        return "neutral"

    def run(self, texts):
        if self.pipe is None:
            scores = [self._rule_score(t) for t in texts]
            labels = [self._label_from_score(s) for s in scores]
            return labels, scores
        out = self.pipe(texts, truncation=True, batch_size=32)
        labels, scores = [], []
        for r in out:
            lab = r["label"].lower()
            if lab.startswith("neg"): lab = "negative"
            elif lab.startswith("pos"): lab = "positive"
            else: lab = "neutral"
            labels.append(lab); scores.append(float(r.get("score",0.0)))
        return labels, scores

# -------------------- Fake Detection --------------------
def compute_embeddings(texts):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        E = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return E, "st"
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vect = TfidfVectorizer(min_df=2, max_df=0.7, ngram_range=(1,2), lowercase=True)
        X = vect.fit_transform(texts)
        from sklearn.preprocessing import normalize
        E = normalize(X, axis=1)
        return E, "tfidf"

def cluster_near_duplicates(E, backend="st"):
    from sklearn.cluster import DBSCAN
    if backend=="st":
        sim = np.clip(E @ E.T, -1, 1)
        dist = 1 - sim
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        sim = cosine_similarity(E)
        dist = 1 - sim
    clustering = DBSCAN(eps=0.22, min_samples=3, metric="precomputed").fit(dist)
    labels = clustering.labels_
    lab, cnt = np.unique(labels[labels!=-1], return_counts=True)
    susp = set(lab[cnt>=5])
    near_dup = np.array([(cl in susp) for cl in labels], dtype=bool)
    return labels, near_dup

def fake_signals(df):
    text = df["content"].astype(str).fillna("")
    norm = text.apply(normalize_tr)
    dup_exact = norm.duplicated(keep=False)
    def low_info(t):
        if len(t)<8: return True
        if REPEAT_RE.search(t): return True
        letters = sum(ch.isalnum() for ch in t)
        return letters < 0.4*max(1,len(t))
    low_info_flag = text.apply(low_info)
    bot_like = text.apply(lambda s: bool(URL_RE.search(s) or CONTACT_RE.search(s)))
    try:
        E, backend = compute_embeddings(text.tolist())
        labels, dup_near = cluster_near_duplicates(E, backend=backend)
    except Exception:
        labels = np.full(len(df), -1, dtype=int); dup_near = np.zeros(len(df),dtype=bool)
    fake_flag = dup_exact | low_info_flag | bot_like | dup_near
    return fake_flag, dup_exact, dup_near, low_info_flag, bot_like, labels

# -------------------- Interesting --------------------
class InterestingEngine:
    def __init__(self, device=-1):
        self.pipe = None
        self.labels = ["humorous","constructive","exaggerated","suggestive","novel"]
        try:
            from transformers import pipeline
            self.pipe = pipeline("zero-shot-classification",
                                 model="joeddav/xlm-roberta-large-xnli", device=device)
        except Exception:
            self.pipe = None

    def run(self, df):
        txt   = df["content"].astype(str).fillna("")
        likes = safe_series(df,"thumbs_up_count",0).astype(float)
        stars = safe_series(df,"score_stars",0)
        L = txt.str.len().values
        exclam = txt.apply(lambda s: s.count("!")+s.count("?")).values
        emojis = txt.apply(lambda s: sum(ch in s for ch in LAUGH_TOKENS)).values
        sent_int = np.abs(df["sentiment_score"].fillna(0).astype(float).values)
        # zero-shot
        zs_scores = np.zeros((len(df),len(self.labels)))
        if self.pipe:
            out = self.pipe(txt.tolist(), candidate_labels=self.labels, multi_label=True, batch_size=16)
            for i,r in enumerate(out):
                smap={lab.lower():sc for lab,sc in zip(r["labels"],r["scores"])}
                zs_scores[i]=[smap.get(lbl,0.0) for lbl in self.labels]
        z = (0.22*np.tanh(L/220)+0.15*np.tanh(exclam/3)+0.14*np.tanh(likes/10)
             +0.10*np.tanh(emojis/1)+0.13*np.tanh(sent_int/0.6)
             +0.18*np.tanh(zs_scores.sum(axis=1)))
        thr=np.quantile(z,0.70)
        flags=z>=thr
        reasons=[]
        for i in range(len(df)):
            r=[]
            if L[i]>=220: r.append("long")
            if exclam[i]>=3: r.append("expressive")
            if likes.iloc[i]>=10: r.append("community_liked")
            if emojis[i]>0: r.append("humorous")
            if sent_int[i]>0.6: r.append("strong_sentiment")
            if self.pipe is not None:
                for k,lbl in enumerate(self.labels):
                    if zs_scores[i,k]>=0.5: r.append(f"zs_{lbl}")
            reasons.append(",".join(r))
        return flags, z, reasons

# -------------------- Main --------------------
def main():
    assert os.path.isfile(INPUT_CSV), f"Girdi bulunamadı: {INPUT_CSV}"
    df=pd.read_csv(INPUT_CSV)
    for col in ["content","reply"]:
        if col in df.columns: df[col]=df[col].fillna("").astype(str)

    print("▶ Sentiment")
    sent=SentimentEngine()
    df["sentiment_label"], df["sentiment_score"]=sent.run(df["content"].tolist())

    print("▶ Fake detection")
    f, de, dn, li, bl, cid=fake_signals(df)
    df["fake_flag"]=f; df["dup_exact"]=de; df["dup_near"]=dn
    df["low_info"]=li; df["bot_like"]=bl; df["cluster_id"]=cid

    print("▶ Interesting detection")
    intr=InterestingEngine()
    fl, sc, rs=intr.run(df)
    df["interesting_flag"]=fl; df["interesting_score"]=sc; df["interesting_reason"]=rs

    df.to_csv(OUT_CSV,index=False,encoding="utf-8")
    with open(OUT_JSONL,"w",encoding="utf-8") as f:
        for _,row in df.iterrows():
            f.write(json.dumps(row.to_dict(),ensure_ascii=False)+"\n")

    print(f"✅ Çıktı: {OUT_CSV}, {OUT_JSONL}")

if __name__=="__main__":
    main()
