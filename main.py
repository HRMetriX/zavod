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

def generate_post_with_llm(title, summary):
    """Генерация поста через Hugging Face Inference API"""
    PROMPT_TEMPLATE = """
Ты — Витёк из гаража: мужик 50+, бывший заводчанин, с лёгкой контузией после китайского крана. Ты пересказываешь политические новости так, будто услышал их от Сан Саныча у ларька или тёти Любы на лавке. Не называй политиков официально — используй прозвища: Путин = Батенька, Лавров = Слоняра, Си = Китаец с рынка и т.д. Переводи санкции, Совбез, Минобороны на язык быта: "санкции = пыль с Запада", "Совбез = диспетчерская по базару". Начинай пост с живой сцены (лавка, огород, пивной ларёк...), добавляй детали вроде "водка подорожала", "у меня огурцы солить", "Петрович, драть его в сраку!". Заканчивай иронично: "А мне-то чё? У меня гараж есть". Подпись: "За Родину-мать не стыдно рвать!" 🇷🇺.

Вот новость: "{title}. {summary}"

Напиши пост в стиле Витёка. В конце добавь строку:  
ПРОМПТ ДЛЯ КАРТИНКИ: [описание сцены в стиле российской провинции, с юмором, абсурдом, деталями, без текста на изображении].
"""
    prompt = PROMPT_TEMPLATE.format(title=title, summary=summary)

    API_URL = "https://api-inference.huggingface.co/models/IlyaGusev/saiga_llama3_8b"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "temperature": 0.9,
            "do_sample": True,
            "return_full_text": False
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise Exception(f"HF error: {response.text}")
    
    generated = response.json()[0]["generated_text"]
    return generated.strip()


def generate_image_with_kandinsky(prompt):
    """Генерация картинки через Kandinsky 3.1 (Fusion Brain API)"""
    url = "https://api.fusionbrain.ai/api/v1/text2image"
    headers = {
        "X-Key": FB_API_KEY,
        # "X-Secret": FB_SECRET_KEY,  # часто не требуется — закомментируй, если не используешь
        "Content-Type": "application/json"
    }
    payload = {
        "model_id": "7582",  # Kandinsky 3.1
        "params": {
            "prompt": prompt + ", russian provincial town, humorous, detailed, no text, no letters",
            "negative_prompt": "blurry, ugly, text, signature, watermark, deformed",
            "width": 1024,
            "height": 1024,
            "steps": 30,
            "seed": random.randint(1, 1000000)
        }
    }

    response = requests.post(url, headers=headers, json=payload, timeout=120)
    if response.status_code != 200:
        raise Exception(f"Kandinsky error: {response.text}")
    
    image_url = response.json()["result"][0]["image_url"]
    img_data = requests.get(image_url).content
    img_path = "/tmp/vitok_post.jpg"
    with open(img_path, "wb") as f:
        f.write(img_data)
    return img_path


def send_to_telegram(text, image_path=None):
    """Отправка в Telegram"""
    # Убираем пробелы в URL!
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    
    # Отправка текста
    data = {
        "chat_id": CHANNEL,
        "text": text[:4096],  # Telegram limit
        "parse_mode": "HTML"
    }
    requests.post(f"{base_url}/sendMessage", data=data)

    # Отправка картинки (если есть)
    if image_path:
        try:
            with open(image_path, "rb") as img:
                files = {"photo": img}
                data = {"chat_id": CHANNEL}
                requests.post(f"{base_url}/sendPhoto", files=files, data=data)
        except Exception as e:
            print(f"⚠️ Не удалось отправить картинку: {e}")


# === MAIN ===
if __name__ == "__main__":
    print("🔍 Ищу свежие политические новости...")
    news = fetch_political_news(hours=1)

    if not news:
        print("😴 Нет свежих новостей за последний час.")
        save_seen()
        exit(0)

    item = news[0]
    print(f"📰 Нашёл: {item['title']}")

    full_output = ""
    try:
        print("🧠 Генерирую пост через LLM...")
        full_output = generate_post_with_llm(item["title"], item["summary"])

        # Извлекаем текст и промпт для картинки
        if "ПРОМПТ ДЛЯ КАРТИНКИ:" in full_output:
            text_part, img_prompt_raw = full_output.split("ПРОМПТ ДЛЯ КАРТИНКИ:", 1)
            text = text_part.strip()
            img_prompt = img_prompt_raw.strip().strip("[]\"' ")
        else:
            text = full_output
            img_prompt = "A typical Russian provincial town, a man on a bench reading news, beer bottle nearby, humorous cartoon style"

        print("🎨 Генерирую картинку...")
        img_path = generate_image_with_kandinsky(img_prompt)

        print("📤 Постим в Telegram...")
        send_to_telegram(text, img_path)

        print("✅ Успешно опубликовано!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        # Отправляем текст даже при ошибке генерации картинки
        fallback_text = f"[⚠️ Ошибка в генерации]\n\n{full_output[:4000] if full_output else item['title']}"
        send_to_telegram(fallback_text)

    save_seen()
