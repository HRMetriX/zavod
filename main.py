import os
import json
import time
import random
import feedparser
from datetime import datetime, timedelta
import requests
from openai import OpenAI
import traceback
from huggingface_hub import InferenceClient


# === CONFIG ===
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
HF_TOKEN = os.environ["HF_TOKEN"]
FB_API_KEY = os.environ["FUSIONBRAIN_API_KEY"]
GIST_TOKEN = os.environ["GIST_TOKEN"]
CHANNEL = os.environ.get("TELEGRAM_CHANNEL", "@notreviews")

RSS_SOURCES = [
    "https://ria.ru/export/rss2/archive/index.xml",
    "https://tass.ru/rss/v2.xml",
    "https://lenta.ru/rss/",
]

# === GIST STATE MANAGEMENT ===
GIST_ID = "5944017a021bcea90b63cf408a0324e5"

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
        requests.patch(url, headers=headers, json=payload, timeout=10)
        print("✅ seen.json обновлён в Gist")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения в Gist: {e}")

# === NEWS PARSING ===
def fetch_political_news(hours=1):
    keywords = ["Путин", "президент", "Совбез", "Минобороны", "Лавров", "Шойгу", "назнач", "указ", "Санчик", "Булыга", "Россия", "политик", "Си", "Зеленск", "Байден", "Трамп"]
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

# === LLM ===
def generate_post_with_llm(title, summary):
    """Генерация поста через Hugging Face Inference Providers (Qwen2.5 via Together)"""
    PROMPT_TEMPLATE = """
Ты — Витёк из гаража: мужик 50+, бывший заводчанин, с лёгкой контузией после китайского крана. Ты пересказываешь политические новости так, будто услышал их от Сан Саныча у ларька или тёти Любы на лавке. Не называй политиков официально — используй прозвища: Путин = Батенька, Лавров = Слоняра, Си = Китаец с рынка и т.д. Переводи санкции, Совбез, Минобороны на язык быта: "санкции = пыль с Запада", "Совбез = диспетчерская по базару". Начинай пост с живой сцены (лавка, огород, пивной ларёк...), добавляй детали вроде "водка подорожала", "у меня огурцы солить", "Петрович, драть его в сраку!". Заканчивай иронично: "А мне-то чё? У меня гараж есть". Подпись: "За Родину-мать не стыдно рвать!" 🇷🇺.

Вот новость: "{title}. {summary}"

Напиши пост в стиле Витёка. В конце добавь строку:  
ПРОМПТ ДЛЯ КАРТИНКИ: [описание сцены в стиле российской провинции, с юмором, абсурдом, деталями, без текста на изображении].
"""
    prompt = PROMPT_TEMPLATE.format(title=title, summary=summary)

    print("📝 Отправляю промпт в Qwen2.5-7B через прямое API...")
    
    # Используем тот же клиент, что работал в тестах
    client = InferenceClient(token=os.environ["HF_TOKEN"])
    
    try:
        # Используем chat_completion с явным указанием модели
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.9
        )
        result = response.choices[0].message.content.strip()
        print("✅ LLM ответил успешно")
        return result

    except Exception as e:
        print(f"❌ Ошибка в InferenceClient: {e}")
        # Fallback на прямое API
        print("🔄 Пробую прямое API...")
        return generate_post_with_llm_fixed(title, summary)


# === KANDINSKY ===
def generate_image_with_kandinsky(prompt):
    url = "https://api.fusionbrain.ai/api/v1/text2image"
    headers = {
        "X-Key": FB_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "model_id": "7582",
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

# === TELEGRAM ===
def send_to_telegram(text, image_path=None):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    data = {
        "chat_id": CHANNEL,
        "text": text[:4096],
        "parse_mode": "HTML"
    }
    requests.post(f"{base_url}/sendMessage", data=data)

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
    print("🔍 Загружаем уже обработанные новости из Gist...")
    seen_titles = load_seen()

    print("🔍 Ищу свежие политические новости...")
    news = fetch_political_news(hours=1)

    if not news:
        print("😴 Нет свежих новостей за последний час.")
        save_seen(seen_titles)
        exit(0)

    item = news[0]
    print(f"📰 Нашёл: {item['title']}")

    full_output = ""
    try:
        print("🧠 Генерирую пост через LLM...")
        full_output = generate_post_with_llm(item["title"], item["summary"])

        if "ПРОМПТ ДЛЯ КАРТИНКИ:" in full_output:
            text_part, img_prompt_raw = full_output.split("ПРОМПТ ДЛЯ КАРТИНКИ:", 1)
            text = text_part.strip()
            img_prompt = img_prompt_raw.strip().strip("[]\"' ")
        else:
            text = full_output
            img_prompt = "A Russian man on a bench in a small town, reading news, beer bottle nearby, humorous style"

        print("🎨 Генерирую картинку...")
        img_path = generate_image_with_kandinsky(img_prompt)

        print("📤 Постим в Telegram...")
        send_to_telegram(text, img_path)

        print("✅ Успешно опубликовано!")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        fallback_text = f"[⚠️ Ошибка в генерации]\n\n{full_output[:4000] if full_output else item['title']}"
        send_to_telegram(fallback_text)

    save_seen(seen_titles)

print("🏁 Скрипт завершён. Всего обработано новостей:", len(news))
