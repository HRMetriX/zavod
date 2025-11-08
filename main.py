# main.py
import os
import json
import time
import random
import feedparser
from datetime import datetime, timedelta
import requests

# === CONFIG ===
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
HF_TOKEN = os.environ["HF_TOKEN"]
FB_API_KEY = os.environ["FUSIONBRAIN_API_KEY"]
FB_SECRET_KEY = os.environ.get("FUSIONBRAIN_SECRET_KEY", "")  # может не понадобиться
CHANNEL = os.environ.get("TELEGRAM_CHANNEL", "@your_channel_here")  # или "@mychannel"

RSS_SOURCES = [
    "https://ria.ru/export/rss2/archive/index.xml",
    "https://tass.ru/rss/v2.xml",
    "https://lenta.ru/rss/",
]

# Загружаем уже обработанные заголовки
if os.path.exists("seen.json"):
    with open("seen.json", "r") as f:
        seen_titles = set(json.load(f))
else:
    seen_titles = set()

def save_seen():
    with open("seen.json", "w") as f:
        json.dump(list(seen_titles), f, ensure_ascii=False)

def fetch_political_news(hours=1):
    """Парсим свежие новости и фильтруем по ключевым словам"""
    keywords = ["Путин", "президент", "Совбез", "Минобороны", "Лавров", "Шойгу", "назнач", "указ", "Санчик", "Булыга", "Россия", "политик"]
    fresh = []
    cutoff = datetime.now() - timedelta(hours=hours)

    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                pub = datetime(*entry.published_parsed[:6])
                if pub < cutoff:
                    continue
                title = entry.title
                if title in seen_titles:
                    continue
                # Фильтр по политике
                if any(kw in title or kw in entry.get("summary", "") for kw in keywords):
                    fresh.append({
                        "title": title,
                        "summary": entry.get("summary", "")[:300],
                        "link": entry.link
                    })
                    seen_titles.add(title)
        except Exception as e:
            print(f"Ошибка парсинга {url}: {e}")
    
    return fresh

