from pathlib import Path
import subprocess
from faster_whisper import WhisperModel


INPUT_FOLDER = Path("input")
OUTPUT_FOLDER = Path("output")
OUTPUT_AUDIO = OUTPUT_FOLDER / "audio.wav"
OUTPUT_TRANSCRIPT = OUTPUT_FOLDER / "transcript.txt"


def find_input_video() -> Path | None:
    """
    Ищет первый подходящий видеофайл в папке input.
    """
    video_extensions = [".mp4", ".mov", ".m4v", ".avi", ".mkv"]

    for file in INPUT_FOLDER.iterdir():
        if file.suffix.lower() in video_extensions:
            return file

    return None


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


def transcribe_audio(audio_path: Path, transcript_path: Path) -> None:
    """
    Расшифровывает аудио в текст через faster-whisper.
    """
    print("Загружаю модель распознавания речи...")

    model = WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8"
    )

    print("Расшифровываю аудио...")

    segments, info = model.transcribe(
        str(audio_path),
        language="ru",
        beam_size=5
    )

    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write("РАСШИФРОВКА РОЛИКА\n")
        file.write("=" * 40 + "\n\n")
        file.write(f"Язык: {info.language}\n")
        file.write(f"Вероятность языка: {round(info.language_probability, 2)}\n\n")

        for segment in segments:
            start = round(segment.start, 2)
            end = round(segment.end, 2)
            text = segment.text.strip()

            file.write(f"[{start} - {end}] {text}\n")

    print(f"Расшифровка сохранена: {transcript_path}")


def main() -> None:
    print("ReelsAssistant запущен")

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    video_path = find_input_video()

    if video_path is None:
        print("Ошибка: видео не найдено в папке input")
        print("Положи видео в папку input")
        print("Поддерживаемые форматы: .mp4, .mov, .m4v, .avi, .mkv")
        return

    print(f"Найдено видео: {video_path}")

    print("1. Вытаскиваю аудио из видео...")
    extract_audio(video_path, OUTPUT_AUDIO)

    print("2. Делаю расшифровку...")
    transcribe_audio(OUTPUT_AUDIO, OUTPUT_TRANSCRIPT)

    print("Готово")


if __name__ == "__main__":
    main()
