# scrape_patrol_officer_full.py
# Gerekenler:
#   pip install google-play-scraper pandas pyarrow tqdm langdetect

import os, time
import pandas as pd
from tqdm import tqdm
from langdetect import detect as _detect
from google_play_scraper import reviews_all

# ---- OYUN PAKETİ ----
PKG = "com.flatgames.patrolofficer"

# ---- KAPSAM ----
LANGS = [
    "tr","en","ar","ru","de","fr","es","pt","it","ja","ko","zh","hi","id","vi","th",
    "pl","nl","sv","da","fi","no","el","he","cs","sk","ro","hu","uk","bg","sr","hr","sl",
    "ms","fil","fa","ur","bn","ta","te","ml","mr","gu","kn","pa","si","km","lo","my","ne",
    "am","az","be","bs","ca","et","eu","gl","hy","is","ka","kk","ky","lt","lv","mk","mn",
    "ps","sq","sw","uz"
]
COUNTRIES = [
    "TR","US","GB","DE","FR","ES","IT","PT","BR","MX","AR","CO","CL","PE","CA","AU","NZ","IN",
    "ID","MY","PH","TH","VN","JP","KR","TW","HK","NL","BE","SE","NO","DK","FI","PL","CZ","SK",
    "RO","HU","GR","UA","KZ","SA","AE","EG","MA","ZA","IL"
]

# ---- ÇIKTILAR ----
BASE_DIR    = "data/bronze/pkg=com.flatgames.patrolofficer"  # Parquet parçaları
OUT_CSV     = "patrol_officer_reviews.csv"                   # opsiyonel toplu CSV
CHECKPOINT  = "patrol_officer_ids.txt"                       # tekilleştirme/resume

# ------------------------------------------------------------

def lang_detect_safe(t: str) -> str:
    try:
        if not t or len(t.strip()) < 6:
            return ""
        return _detect(t.strip())
    except Exception:
        return ""

def write_parquet_part(df: pd.DataFrame, lang: str, country: str):
    if df.empty: return
    date_str = time.strftime("%Y-%m-%d")
    out_dir = os.path.join(BASE_DIR, f"date_ingested={date_str}", f"lang={lang}", f"country={country}")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"part-{int(time.time()*1000)}.parquet"
    df.to_parquet(os.path.join(out_dir, fname), index=False)

def write_csv_append(df: pd.DataFrame):
    if df.empty: return
    header = not os.path.isfile(OUT_CSV)
    df.to_csv(OUT_CSV, mode="a", index=False, header=header)

def main():
    # checkpoint yükle
    seen = set()
    if os.path.isfile(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            for line in f:
                seen.add(line.strip())

    total_new = 0
    for lang in tqdm(LANGS, desc="langs"):
        for country in COUNTRIES:
            try:
                batch = reviews_all(PKG, lang=lang, country=country)
            except Exception as e:
                print(f"[WARN] {lang}-{country}: {e}")
                continue

            rows = []
            for r in batch:
                rid = r.get("reviewId")
                if not rid or rid in seen:
                    continue
                seen.add(rid)
                rows.append({
                    "review_id": rid,
                    "user_name": r.get("userName"),
                    "score_stars": r.get("score"),
                    "thumbs_up_count": r.get("thumbsUpCount", 0),
                    "at": r.get("at").isoformat() if r.get("at") else None,
                    "content": r.get("content") or "",
                    "reply": r.get("replyContent") or "",
                    "lang_param": lang,
                    "country_param": country,
                    "lang_detected": lang_detect_safe(r.get("content") or "")
                })

            if rows:
                df = pd.DataFrame(rows)
                write_parquet_part(df, lang, country)
                write_csv_append(df)
                total_new += len(df)
                with open(CHECKPOINT, "a", encoding="utf-8") as f:
                    for _id in df["review_id"].tolist():
                        f.write(str(_id) + "\n")

    print(f"DONE → yeni kayıt: {total_new}")
    print(f"Parquet base: {BASE_DIR}")
    print(f"CSV: {OUT_CSV if os.path.isfile(OUT_CSV) else 'yok'}")
    print(f"Checkpoint: {CHECKPOINT}")

if __name__ == "__main__":
    main()
