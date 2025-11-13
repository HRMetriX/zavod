import os
import json
import time
import random
import feedparser
from datetime import datetime, timedelta
import requests
from huggingface_hub import InferenceClient
import base64

def load_seen():
    url = f"https://api.github.com/gists/{GIST_ID}"
    headers = {"Authorization": f"token {GIST_TOKEN}"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            files = resp.json().get("files", {})
            if "seen.json" in files:
                content = files["seen.json"].get("content", "[]")
                return set(json.loads(content))
        return set()
    except Exception as e:
        print(f"⚠️ Ошибка загрузки seen.json из Gist: {e}")
        return set()

def save_seen(seen_set):
    url = f"https://api.github.com/gists/{GIST_ID}"
    headers = {"Authorization": f"token {GIST_TOKEN}"}
    payload = {
        "files": {
            "seen.json": {
                "content": json.dumps(list(seen_set), ensure_ascii=False, indent=2)
            }
        }
    }
    try:
        resp = requests.patch(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            print("✅ seen.json обновлён в Gist")
        else:
            print(f"❌ Ошибка сохранения в Gist: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения в Gist: {e}")

# === NEWS PARSING ===
def fetch_political_news(hours=1):
    keywords = [
    # Политика
    "политик", "указ", "назнач", "Совбез", "Минобороны", "президент", "выборы", "парламент", "госдума", "сенат",
    # Фамилии лидеров
    "Путин", "Лавров", "Шойгу", "Си", "Зеленский", "Байден", "Трамп", "Ким", "Меркель", "Макрон", "Додик", "Медведев", "Володин",
    # Страны
    "Россия", "США", "Китай", "Украина", "Северная Корея", "Европа", "ЕС", "НАТО", "ООН", "ОПЕК", "Беларусь", "Казахстан", "Турция", "Иран", "Израиль", "Палестина", "Сирия", "Афганистан",
    # Организации
    "ООН", "ЕС", "НАТО", "ОПЕК", "МАГАТЭ", "СНГ", "БРИКС", "ШОС", "G7", "G20",
    # Финансы
    "санкции", "биржа", "валюта", "доллар", "евро", "золото", "нефть", "газ", "торги", "рынок", "процент", "ставка", "дефицит", "инфляция", "долг", "кредит"
]
    fresh = []
    cutoff = datetime.now() - timedelta(hours=hours)

    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url.strip())
            for entry in feed.entries:
                pub = datetime(*entry.published_parsed[:6])
                if pub < cutoff:
                    continue
                title = entry.title
                summary = entry.get("summary", "")
                if title in seen_titles:
                    continue
                if any(kw in title or kw in summary for kw in keywords):
                    fresh.append({
                        "title": title,
                        "summary": summary[:300],
                        "link": entry.link
                    })
                    seen_titles.add(title)
        except Exception as e:
            print(f"Ошибка парсинга {url}: {e}")
    return fresh
