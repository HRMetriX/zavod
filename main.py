import os
import json
import time
import random
import feedparser
from datetime import datetime, timedelta
import requests
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
        resp = requests.patch(url, headers=headers, json=payload, timeout=10)
        if resp.status_code == 200:
            print("✅ seen.json обновлён в Gist")
        else:
            print(f"❌ Ошибка сохранения в Gist: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Ошибка сохранения в Gist: {e}")

# === NEWS PARSING ===
def fetch_political_news(hours=1):
    keywords = ["Путин", "президент", "Совбез", "Минобороны", "Лавров", "Шойгу", "назнач", "указ", "Санчик", "Булыга", "Россия", "политик", "Си", "Зеленск", "Байден", "Трамп"]
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

# === LLM ===
def generate_post_with_llm(title, summary):
    """Генерация поста через Hugging Face Inference API (Qwen2.5 via Together)"""
    PROMPT_TEMPLATE = """
Ты — Витёк из захолустья: мужик за 50, бывший заводчанин из захолустного городка, где до районного центра на «ПАЗике» два часа ехать, а интернет ловит раз в два дня; у тебя лёгкая контузия после падения на тебя китайского крана на заводе, и ты всё ещё думаешь, что Китай — это один большой рынок за Уралом. Ты пересказываешь реальные политические новости так, будто только что услышал их от Сан Саныча у ларька, тёти Любы на лавке или в очереди за водкой: не называй никого официально — Батенька вместо Путина, Слоняра вместо Лаврова, Китаец с рынка вместо Си Цзиньпина, Клоун с кукурузника вместо Зеленского, НАТО — бандитская шайка с Запада, Совбез ООН — диспетчерская по базару, санкции — пыль с Запада или ветер в жопу от Европы, Минобороны — армейский склад под замком; начинай пост с живой бытовой сцены — огород, гараж, очередь за хлебом, пивной ларёк — и добавляй детали вроде «водка опять подорожала», «у меня огурцы солить», «Петрович, драть его в сраку!»; суть новости при этом должна оставаться узнаваемой — читатель должен понять, кто что сделал и какие последствия, даже под слоем сатиры; заканчивай иронично и отстранённо: «А мне-то чё? У меня гараж есть»; подпись: «За Родину-мать не стыдно рвать!» 🇷🇺.

ВАЖНО: Отвечай ТОЛЬКО на РУССКОМ языке, без использования других языков, особенно китайского.
ВАЖНО: Текст должен быть удобен для чтения, разбит на абзацы, диалоги оформлены удобно.
КРИТИЧНО: Старайся уместить текст в 1000 символов

Вот новость: "{title}. {summary}"

Напиши пост в стиле Витька. В конце добавь строку:  
ПРОМПТ ДЛЯ КАРТИНКИ: [описание сцены в стиле российской провинции, с юмором, абсурдом, деталями, без текста на изображении].
"""
    prompt = PROMPT_TEMPLATE.format(title=title, summary=summary)

    print("📝 Отправляю промпт в Qwen3-235B...")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN не найден!")
        # Возвращаем ТОЛЬКО текст, БЕЗ промпта для картинки
        fallback_text = f"Батенька опять в новостях: {title}. А мне-то чё? У меня гараж есть. За Родину-мать не стыдно рвать! 🇷🇺"
        return fallback_text

    client = InferenceClient(token=hf_token)

    try:
        response = client.chat_completion(
            model="Qwen/Qwen3-235B-A22B-Instruct-2507", #Qwen2.5-7B-Instruct (точно работает), Qwen3-235B-A22B-Instruct-2507-FP8 (не работает)
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7
        )
        result = response.choices[0].message.content.strip()
        print("✅ LLM ответил успешно")
        return result

    except Exception as e:
        print(f"❌ Ошибка в LLM: {e}")
        # Возвращаем ТОЛЬКО текст, БЕЗ промпта для картинки
        fallback_text = f"Батенька опять в новостях: {title}. А мне-то чё? У меня гараж есть. За Родину-мать не стыдно рвать! 🇷🇺"
        return fallback_text
        
# === KANDINSKY ===
def generate_image_with_kandinsky(prompt):
    """
    Генерация изображения через FusionBrain API (асинхронно)
    """
    # URL и заголовки
    base_url = "https://api-key.fusionbrain.ai/"
    api_key = os.environ.get("FUSIONBRAIN_API_KEY")
    secret_key = os.environ.get("FUSIONBRAIN_SECRET_KEY")

    if not api_key:
        print("❌ FUSIONBRAIN_API_KEY не найден в переменных окружения")
        return None
    if not secret_key:
        print("❌ FUSIONBRAIN_SECRET_KEY не найден в переменных окружения")
        return None

    headers = {
        'X-Key': f'Key {api_key}',
        'X-Secret': f'Secret {secret_key}',
    }

    # 1. Получаем список доступных моделей (pipeline_id)
    try:
        resp = requests.get(base_url + 'key/api/v1/pipelines', headers=headers)
        if resp.status_code != 200:
            print(f"❌ Ошибка получения списка моделей: {resp.status_code}, {resp.text}")
            return None
        pipelines = resp.json()
        if not pipelines:
            print("❌ Нет доступных моделей")
            return None
        # Берём первую доступную модель (обычно это Kandinsky 3.1)
        pipeline_id = pipelines[0]['id']
        print(f"✅ Используем модель: {pipelines[0]['name']} (ID: {pipeline_id})")
    except Exception as e:
        print(f"❌ Ошибка при получении pipeline_id: {e}")
        return None

    # 2. Подготовим параметры для генерации (правильная структура!)
    params = {
        "type": "GENERATE",
        "numImages": 1,
        "width": 1024,
        "height": 1024,
        "negativePromptDecoder": "blurry, ugly, text, signature, watermark, deformed",  # <-- ВНЕ generateParams
        "generateParams": {
            "query": prompt + ", russian provincial town, humorous, detailed, no text, no letters",
        }
    }

    # 3. Отправляем задачу на генерацию (multipart/form-data)
    data = {
        'pipeline_id': (None, pipeline_id),
        'params': (None, json.dumps(params), 'application/json')
    }

    try:
        resp = requests.post(base_url + 'key/api/v1/pipeline/run', headers=headers, files=data)
        # ИСПРАВЛЕНО: 201 — это успех, а не ошибка
        if resp.status_code != 201:
            print(f"❌ Ошибка при отправке задачи: {resp.status_code}, {resp.text}")
            return None
        result = resp.json()
        if 'uuid' not in result:
            print(f"❌ Ошибка: нет uuid в ответе: {result}")
            return None
        uuid = result['uuid']
        print(f"✅ Задача отправлена, UUID: {uuid}")
    except Exception as e:
        print(f"❌ Ошибка при отправке задачи на генерацию: {e}")
        return None

    # 4. Ждём завершения генерации
    attempts = 20  # <-- Увеличено
    delay = 15  # <-- Увеличено (15 секунд)
    print(f"⏳ Ожидание генерации... (до {attempts * delay} секунд)")
    while attempts > 0:
        try:
            resp = requests.get(base_url + f'key/api/v1/pipeline/status/{uuid}', headers=headers)
            if resp.status_code != 200:
                print(f"❌ Ошибка проверки статуса: {resp.status_code}, {resp.text}")
                return None
            status_data = resp.json()

            if status_data['status'] == 'DONE':
                print("✅ Генерация завершена!")
                image_url = status_data['result']['files'][0]
                img_data = requests.get(image_url).content
                img_path = "/tmp/vitok_post.jpg"
                with open(img_path, "wb") as f:
                    f.write(img_data)
                print(f"✅ Изображение сохранено: {img_path}")
                return img_path
            elif status_data['status'] == 'FAILED':
                print(f"❌ Генерация изображения не удалась: {status_data.get('errorDescription', 'Unknown error')}")
                return None
            else:
                print(f"⏳ Статус: {status_data['status']}, ожидание...")

        except Exception as e:
            print(f"❌ Ошибка при проверке статуса: {e}")
            return None

        attempts -= 1
        time.sleep(delay)

    print("❌ Превышено время ожидания генерации")
    return None

# === TELEGRAM ===
def send_to_telegram(text, image_path=None):
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    data = {
        "chat_id": CHANNEL,
        "text": text[:4096],
        "parse_mode": "HTML"
    }
    resp = requests.post(f"{base_url}/sendMessage", data=data)

    if image_path and resp.status_code == 200:
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
