import os
os.environ["PYANNOTE_DONT_USE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYANNOTE_CACHE"] = os.path.expanduser("~/.cache/torch/pyannote")

import whisper
import ffmpeg
import srt
import datetime
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from pyannote.audio import Pipeline


load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


pipeline = None
if HUGGINGFACE_TOKEN:
    try:
        login(HUGGINGFACE_TOKEN)

        snapshot_download("pyannote/speaker-diarization", token=HUGGINGFACE_TOKEN, local_dir=os.environ["PYANNOTE_CACHE"])
        snapshot_download("speechbrain/spkrec-ecapa-voxceleb", token=HUGGINGFACE_TOKEN, local_dir=os.environ["PYANNOTE_CACHE"])

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
            use_auth_token=HUGGINGFACE_TOKEN,
            cache_dir=os.environ["PYANNOTE_CACHE"]
        )

        print("✅ Speaker diarization pipeline loaded successfully")
    except Exception as e:
        print(f"❌ Pipeline loading failed: {e}")
else:
    print("❌ Hugging Face token not found")


def transcribe_and_diarize(audio_path: str, language_code: str = "en") -> str:
    """Transcribes and diarizes the input audio, returns path to SRT file"""
    if not pipeline:
        raise RuntimeError("Diarization pipeline unavailable")

    wav_path = os.path.splitext(audio_path)[0] + "_converted.wav"
    srt_path = os.path.splitext(audio_path)[0] + ".srt"

    try:
        
        (
            ffmpeg.input(audio_path)
            .output(wav_path, ar=16000, ac=1)
            .run(overwrite_output=True, quiet=True)
        )

        
        diarization = pipeline(wav_path, num_speakers=2)

        
        model = whisper.load_model("large")  
        result = model.transcribe(wav_path, language=language_code, word_timestamps=True)

        
        segments = []
        speakers = {}
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speakers:
                speakers[speaker] = f"Speaker {len(speakers) + 1}"

            words = []
            for seg in result["segments"]:
                for word in seg.get("words", []):
                    if turn.start <= word["start"] and word["end"] <= turn.end:
                        words.append(word["word"])

            text = " ".join(words)

            if text:
                segments.append(srt.Subtitle(
                    index=len(segments) + 1,
                    start=datetime.timedelta(seconds=turn.start),
                    end=datetime.timedelta(seconds=turn.end),
                    content=f"[{speakers[speaker]}] {text}"
                ))


        
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt.compose(segments))

        print(f"✅ Subtitle saved at: {srt_path}")
        return srt_path

    except Exception as e:
        raise RuntimeError(f"Processing failed: {e}")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
