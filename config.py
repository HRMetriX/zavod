import os

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

GIST_ID = "5944017a021bcea90b63cf408a0324e5"
