
import os
import re
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from googleapiclient.discovery import build

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_COMMENTS = 200

if not YOUTUBE_API_KEY:
    raise RuntimeError("Missing YOUTUBE_API_KEY environment variable.")

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

app = FastAPI(title="YouTube Sentiment Analyzer")

class UrlPayload(BaseModel):
    url: str

@app.get("/")
def home():
    return FileResponse("index.html")

def extract_video_id(url: str):
    m = re.search(r"v=([a-zA-Z0-9_-]{8,})", url)
    if m: return m.group(1)
    m = re.search(r"youtu\.be/([a-zA-Z0-9_-]{8,})", url)
    if m: return m.group(1)
    return None

def get_youtube_comments(video_id: str, max_comments: int = MAX_COMMENTS):
    comments = []
    req = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
    )

    response = req.execute()

    while response and len(comments) < max_comments:
        for item in response.get("items", []):
            try:
                comments.append(item["snippet"]["topLevelComment"]["snippet"]["textDisplay"])
            except:
                continue

        if "nextPageToken" in response:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response["nextPageToken"],
                textFormat="plainText"
            ).execute()
        else:
            break

    return comments[:max_comments]

def analyze_text(text: str):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    pos = float(probs[1])
    neg = float(probs[0])

    if 0.45 < pos < 0.55:
        sentiment = "Neutral"
    else:
        sentiment = "Positive" if pos > neg else "Negative"

    return {
        "text": text,
        "sentiment": sentiment,
        "confidence_positive": pos,
        "confidence_negative": neg
    }

def analyze_all(comments: List[str]):
    summary = {"positive": 0, "negative": 0, "neutral": 0}
    detailed = []

    for c in comments:
        result = analyze_text(c)
        summary[result["sentiment"].lower()] += 1
        detailed.append(result)

    return summary, detailed

@app.post("/analyze")
def analyze(payload: UrlPayload):
    video_id = extract_video_id(payload.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    comments = get_youtube_comments(video_id)
    summary, detailed = analyze_all(comments)

    return {
        "video_id": video_id,
        "total_comments": len(comments),
        "summary": summary,
        "comments": detailed
    }
