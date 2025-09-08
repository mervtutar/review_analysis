# Google Play Review Analysis 

This project first collects all reviews for the game from Google Play, then automatically analyzes each review for sentiment (positive / neutral / negative), detects whether it’s likely fake (bot, spam, duplicate, or fabricated), and scores how “interesting” the comment is. Technically, reviews are pulled into a CSV using `google-play-scraper`; sentiment uses a multilingual model (XLM-RoBERTa); fake detection combines simple rules (links/phones), timing patterns (bursts), and similarity clusters (text embeddings + DBSCAN), with exceptions so natural short praise and genuine “ads/time” complaints aren’t flagged incorrectly; interesting reviews are scored using length, emojis/punctuation, likes, rare wording, and zero-shot labels. Run analysis with `python analyze.py` and open a simple dashboard with `streamlit run dashboard.py`. Thresholds (burst size, duplicate cluster size, etc.) are easy to tweak in the code.

---

## What it does
- Collects all Google Play reviews for the game.  
- Tags each review with sentiment, a fake/not-fake decision, and an **interesting** score.  
- Shows results in a simple Streamlit table and saves them to CSV/JSONL.

---

## How scraping works (simple)
- Uses `google-play-scraper` to pull reviews from many locales.  
- Keeps a checkpoint of seen IDs so it can resume and avoid duplicates.  
- Writes results to CSV (and Parquet parts for storage).

---

## How sentiment is found
- Uses a multilingual model (XLM-RoBERTa) to label reviews **positive / neutral / negative** and give a confidence score.  
- If the model can’t run, a simple keyword fallback is used.

---

## How fake reviews are detected (plain language)
- **Bots:** If a review contains links, phone numbers, or emails it’s suspicious.  
- **Spam (burst):** If the exact same text appears many times in a short window, it’s likely spam.  
- **Duplicates:** If many very similar reviews appear (measured with text embeddings and clustering), that’s suspicious — but very short common praises are ignored so they don’t become false positives.  
- **Fabricated:** If text and metadata contradict (e.g., super-negative text with a 5-star rating) or the text looks unnatural, and there’s another signal, it’s flagged.  
- **Important exception:** Natural complaints like “1 min ads / 15s gameplay” aren’t marked fake just because many people write them similarly.

---

## How “interesting” reviews are found
- Combines easy signals: long text, emojis, many exclamation marks, number of likes, rare words.  
- Also uses a zero-shot classifier to spot tones like **humorous** or **constructive**.  
- Reviews above a score threshold are marked **interesting** and get a short reason label (e.g., `community_liked(12)`).

---

## How to run (quick)
```bash
# Analyze
python analyze.py
# -> creates patrol_officer_reviews_analyzed.csv

# Dashboard (open the UI)
streamlit run dashboard.py
```

<img width="1920" height="1080" alt="Ekran Görüntüsü (2237)" src="https://github.com/user-attachments/assets/1130734a-6c06-41fc-a368-6648b4a58e9b" />
<img width="1920" height="1080" alt="Ekran Görüntüsü (2238)" src="https://github.com/user-attachments/assets/d93c30d9-d04c-4871-a3b9-e1d77484fcaa" />

