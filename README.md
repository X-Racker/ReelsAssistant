# ReelsAssistant

ReelsAssistant — Python-инструмент для работы с видео под Reels / Shorts / TikTok.

Программа берёт видео, вытаскивает из него аудио, делает расшифровку речи и создаёт простой монтажёрский разбор с идеями для хуков, B-roll и субтитров.

## Что умеет

- Принимать видеофайл `.mp4`, `.mov`, `.m4v`, `.avi`, `.mkv`
- Извлекать аудио через FFmpeg
- Делать расшифровку речи через faster-whisper
- Сохранять расшифровку с таймкодами
- Создавать монтажёрский brief:
  - возможные хуки
  - сильные фразы
  - идеи для нарезки
  - идеи для B-roll
  - идеи для субтитров

## Структура проекта

```text
ReelsAssistant/
├── input/
│   └── video.mov
├── output/
│   └── video/
│       ├── audio.wav
│       ├── transcript.txt
│       └── edit_brief.txt
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
