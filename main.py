from pathlib import Path
import subprocess
import argparse
from faster_whisper import WhisperModel


INPUT_FOLDER = Path("input")
OUTPUT_FOLDER = Path("output")

DEFAULT_MODEL = "tiny"


STRONG_MARKERS = [
    "главное",
    "важно",
    "ошибка",
    "проблема",
    "почему",
    "как",
    "никогда",
    "всегда",
    "результат",
    "деньги",
    "клиент",
    "контент",
    "монтаж",
    "внимание",
    "удержание",
    "продажи",
    "бизнес",
    "рост",
    "секрет",
    "фишка",
    "смысл",
]


BROLL_IDEAS = {
    "деньги": "Показать деньги, оплату, чек, график дохода или экран с цифрами.",
    "клиент": "Показать переписку с клиентом, созвон, бриф или рабочий чат.",
    "монтаж": "Показать таймлайн Premiere Pro / After Effects, нарезку кадров, субтитры.",
    "контент": "Показать съёмку, камеру, телефон, Reels/Shorts/TikTok интерфейс.",
    "ошибка": "Добавить красный маркер, glitch, стоп-кадр или визуальное выделение ошибки.",
    "проблема": "Показать человека в затруднении, хаос на рабочем столе, сложную схему.",
    "результат": "Показать финальный ролик, рост просмотров, график или довольного клиента.",
    "бизнес": "Показать офис, встречу, ноутбук, таблицу, CRM или презентацию.",
    "рост": "Показать график роста, стрелку вверх, статистику, цифры.",
    "внимание": "Добавить резкий zoom, sound effect, крупный акцент на слове.",
}


def parse_args() -> argparse.Namespace:
    """
    Читает аргументы из терминала.
    Пример:
    python main.py input/video.mov --model tiny
    """
    parser = argparse.ArgumentParser(
        description="ReelsAssistant — транскрибация видео и создание монтажёрского разбора."
    )

    parser.add_argument(
        "video",
        nargs="?",
        help="Путь к видеофайлу. Например: input/video.mov",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Модель Whisper. tiny — быстро, но хуже. base/small — точнее.",
    )

    return parser.parse_args()


def find_input_video() -> Path | None:
    """
    Ищет первый подходящий видеофайл в папке input.
    Используется, если пользователь не передал видео через аргумент.
    """
    video_extensions = [".mp4", ".mov", ".m4v", ".avi", ".mkv"]

    for file in INPUT_FOLDER.iterdir():
        if file.suffix.lower() in video_extensions:
            return file

    return None


def create_output_paths(video_path: Path) -> dict:
    """
    Создаёт отдельную папку результата под конкретное видео.
    Например:
    input/client.mov -> output/client/
    """
    video_name = video_path.stem
    project_output_folder = OUTPUT_FOLDER / video_name
    project_output_folder.mkdir(parents=True, exist_ok=True)

    return {
        "folder": project_output_folder,
        "audio": project_output_folder / "audio.wav",
        "transcript": project_output_folder / "transcript.txt",
        "brief": project_output_folder / "edit_brief.txt",
        "srt": project_output_folder / "subtitles.srt",
    }


def extract_audio(video_path: Path, audio_path: Path) -> None:
    """
    Вытаскивает аудио из видео через FFmpeg.
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(audio_path),
    ]

    subprocess.run(command, check=True)
    print(f"Аудио сохранено: {audio_path}")


def format_srt_time(seconds: float) -> str:
    """
    Переводит секунды в формат SRT:
    00:00:01,500
    """
    total_milliseconds = int(seconds * 1000)

    hours = total_milliseconds // 3_600_000
    total_milliseconds %= 3_600_000

    minutes = total_milliseconds // 60_000
    total_milliseconds %= 60_000

    secs = total_milliseconds // 1000
    milliseconds = total_milliseconds % 1000

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"

def split_subtitle_text(text: str, max_line_length: int = 42) -> str:
    """
    Разбивает длинный текст субтитра на 1–2 строки,
    чтобы его было удобнее читать в монтажке.
    """
    words = text.split()

    if not words:
        return ""

    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_line_length:
            if current_line:
                current_line += " " + word
            else:
                current_line = word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return "\n".join(lines[:2])

def write_srt(segments: list, srt_path: Path) -> None:
    """
    Создаёт SRT-файл с субтитрами.
    """
    with open(srt_path, "w", encoding="utf-8") as file:
        for index, segment in enumerate(segments, start=1):
            start = format_srt_time(segment.start)
            end = format_srt_time(segment.end)
            text = split_subtitle_text(segment.text.strip())

            file.write(f"{index}\n")
            file.write(f"{start} --> {end}\n")
            file.write(f"{text}\n\n")

    print(f"SRT-субтитры сохранены: {srt_path}")


def transcribe_audio(
    audio_path: Path,
    transcript_path: Path,
    srt_path: Path,
    model_name: str,
) -> None:
    """
    Расшифровывает аудио в текст через faster-whisper
    и дополнительно создаёт SRT-файл.
    """
    print(f"Загружаю модель распознавания речи: {model_name}")

    model = WhisperModel(
        model_name,
        device="cpu",
        compute_type="int8",
    )

    print("Расшифровываю аудио...")

    segments, info = model.transcribe(
        str(audio_path),
        language="ru",
        beam_size=5,
        vad_filter=True,
        initial_prompt=(
            "Это русскоязычное видео для Reels, Shorts или TikTok. "
            "Речь может быть про монтаж, продюсирование, клиентов, "
            "контент, бизнес и социальные сети."
        ),
    )

    segments = list(segments)

    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write("РАСШИФРОВКА РОЛИКА\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Модель: {model_name}\n")
        file.write(f"Язык: {info.language}\n")
        file.write(f"Вероятность языка: {round(info.language_probability, 2)}\n\n")

        for segment in segments:
            start = round(segment.start, 2)
            end = round(segment.end, 2)
            text = segment.text.strip()

            file.write(f"[{start} - {end}] {text}\n")

    print(f"Расшифровка сохранена: {transcript_path}")

    write_srt(segments, srt_path)


def score_phrase(text: str) -> int:
    """
    Даёт фразе баллы, если она похожа на сильный момент для Reels.
    """
    score = 0
    lowered_text = text.lower()

    for marker in STRONG_MARKERS:
        if marker in lowered_text:
            score += 2

    if "?" in text:
        score += 2

    if 30 <= len(text) <= 180:
        score += 1

    if len(text) > 220:
        score -= 1

    return score


def parse_transcript(transcript_path: Path) -> list[dict]:
    """
    Читает transcript.txt и превращает строки с таймкодами в список фраз.
    """
    phrases = []

    with open(transcript_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line.startswith("[") or "]" not in line:
                continue

            time_part, text = line.split("]", 1)
            time_part = time_part.replace("[", "").strip()
            text = text.strip()

            if not text:
                continue

            phrases.append(
                {
                    "time": time_part,
                    "text": text,
                    "score": score_phrase(text),
                }
            )

    return phrases


def find_broll_suggestions(phrases: list[dict]) -> list[str]:
    """
    Ищет, где можно добавить B-roll.
    """
    suggestions = []

    for phrase in phrases:
        text = phrase["text"].lower()

        for keyword, idea in BROLL_IDEAS.items():
            if keyword in text:
                suggestions.append(f"[{phrase['time']}] {idea}")
                break

    return suggestions


def make_subtitle_accent(text: str) -> str:
    """
    Делает простую подсказку, какие слова можно выделить в субтитрах.
    """
    words = text.split()
    accent_words = []

    for word in words:
        clean_word = word.lower().strip(".,!?—-:;")

        if clean_word in STRONG_MARKERS:
            accent_words.append(word.upper())
        else:
            accent_words.append(word)

    return " ".join(accent_words)


def generate_edit_brief(transcript_path: Path, brief_path: Path) -> None:
    """
    Создаёт монтажёрский разбор на основе расшифровки.
    """
    phrases = parse_transcript(transcript_path)

    if not phrases:
        print("Не удалось найти фразы в transcript.txt")
        return

    strong_phrases = sorted(
        phrases,
        key=lambda phrase: phrase["score"],
        reverse=True,
    )

    strong_phrases = [
        phrase for phrase in strong_phrases
        if phrase["score"] > 0
    ]

    hook_candidates = strong_phrases[:5]
    broll_suggestions = find_broll_suggestions(phrases)

    with open(brief_path, "w", encoding="utf-8") as file:
        file.write("МОНТАЖЁРСКИЙ РАЗБОР РОЛИКА\n")
        file.write("=" * 40 + "\n\n")

        file.write("1. ВОЗМОЖНЫЕ ХУКИ ДЛЯ НАЧАЛА\n")
        file.write("-" * 40 + "\n")

        if hook_candidates:
            for phrase in hook_candidates:
                file.write(f"[{phrase['time']}] {phrase['text']}\n")
        else:
            file.write(
                "Сильные хуки автоматически не найдены. "
                "Лучше проверить расшифровку вручную.\n"
            )

        file.write("\n\n2. СИЛЬНЫЕ ФРАЗЫ\n")
        file.write("-" * 40 + "\n")

        if strong_phrases:
            for phrase in strong_phrases[:10]:
                file.write(f"[{phrase['time']}] {phrase['text']}\n")
        else:
            file.write("Сильные фразы автоматически не найдены.\n")

        file.write("\n\n3. ИДЕИ ДЛЯ НАРЕЗКИ\n")
        file.write("-" * 40 + "\n")
        file.write("- Начать ролик с самой сильной фразы, а не обязательно с первой секунды.\n")
        file.write("- Убрать долгие вступления, паузы и слабые объяснения.\n")
        file.write(
            "- Каждые 2–4 секунды добавлять визуальное изменение: "
            "zoom, B-roll, смену плана, акцент в субтитрах.\n"
        )
        file.write(
            "- Если есть вопросительная фраза, её можно использовать "
            "как хук в первые 1–2 секунды.\n"
        )

        file.write("\n\n4. ГДЕ МОЖНО ДОБАВИТЬ B-ROLL\n")
        file.write("-" * 40 + "\n")

        if broll_suggestions:
            for suggestion in broll_suggestions[:12]:
                file.write(f"{suggestion}\n")
        else:
            file.write(
                "Автоматические B-roll подсказки не найдены. "
                "Можно добавить общие кадры: рабочий процесс, "
                "монтажный таймлайн, телефон, статистика, созвон.\n"
            )

        file.write("\n\n5. ИДЕИ ДЛЯ СУБТИТРОВ\n")
        file.write("-" * 40 + "\n")

        if strong_phrases:
            for phrase in strong_phrases[:5]:
                accent_text = make_subtitle_accent(phrase["text"])
                file.write(f"[{phrase['time']}] {accent_text}\n")
        else:
            file.write("Нет фраз для автоматического выделения.\n")

        file.write("\n\n6. ЧЕСТНОЕ ПРЕДУПРЕЖДЕНИЕ\n")
        file.write("-" * 40 + "\n")
        file.write(
            "Этот разбор сделан простой логикой Python, "
            "а не настоящим AI-анализом. Его нужно проверять вручную.\n"
        )

    print(f"Монтажёрский разбор сохранён: {brief_path}")


def main() -> None:
    args = parse_args()

    print("ReelsAssistant запущен")

    if args.video:
        video_path = Path(args.video)
    else:
        video_path = find_input_video()

    if video_path is None:
        print("Ошибка: видео не найдено в папке input")
        print("Передай путь к видео так: python main.py input/video.mov")
        return

    if not video_path.exists():
        print(f"Ошибка: файл не найден: {video_path}")
        return

    output_paths = create_output_paths(video_path)

    print(f"Найдено видео: {video_path}")
    print(f"Папка результата: {output_paths['folder']}")
    print(f"Используемая модель: {args.model}")

    print("1. Вытаскиваю аудио из видео...")
    extract_audio(video_path, output_paths["audio"])

    print("2. Делаю расшифровку...")
    transcribe_audio(
        output_paths["audio"],
        output_paths["transcript"],
        output_paths["srt"],
        args.model,
    )

    print("3. Создаю монтажёрский разбор...")
    generate_edit_brief(output_paths["transcript"], output_paths["brief"])

    print("Готово")


if __name__ == "__main__":
    main()