# Changelog

## [Released]

## [2.0.0] - 2025-11-13

### Added
- Добавлена функция `generate_post_with_llm_v2` с поддержкой системного и пользовательского промптов.
- Добавлена логика хранения истории сгенерированных текстов (`load_history`, `save_history`) в `history.json` для предотвращения повторов.
- Добавлена новая функция `generate_image_with_hf` для генерации изображений через Hugging Face Inference API (Stable Diffusion XL).
- Добавлен файл `CHANGELOG.md` для отслеживания изменений.

### Changed
- Функция `generate_post_with_llm` заменена на новую логику с использованием истории и разделением промптов.
- Механизм генерации изображений изменён: вместо FusionBrain API теперь используется Hugging Face Inference API (`stabilityai/stable-diffusion-xl-base-1.0`).
- В `requirements.txt` добавлена зависимость `Pillow` для работы с изображениями.
- Промпты вынесены в блок #CONFIG в начале файла.

### Removed
- Удалена старая функция `generate_image_with_kandinsky` и связанная с ней логика (асинхронный вызов, `base64`, `requests`, `time`).
