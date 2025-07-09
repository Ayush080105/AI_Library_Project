import os
import uuid
import tempfile
import shutil
import datetime
from pydub import AudioSegment
from google.cloud import storage
from google.cloud import speech_v1p1beta1 as speech


BUCKET_NAME = "ayush_bucket_0716"

def upload_to_gcs(local_path: str, dest_blob_name: str) -> str:
    """Upload file to Google Cloud Storage and return the gs:// URI."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{dest_blob_name}"

def format_srt(response, output_path: str):
    """Generate .srt file from transcription result."""
    index = 1
    with open(output_path, "w", encoding="utf-8") as f:
        for result in response.results:
            alt = result.alternatives[0]
            if not alt.words:
                continue

            start = alt.words[0].start_time.total_seconds()
            end = alt.words[-1].end_time.total_seconds()
            text = alt.transcript.strip()

            def fmt_time(seconds):
                td = datetime.timedelta(seconds=int(seconds))
                millis = int((seconds - int(seconds)) * 1000)
                return f"{str(td)},{millis:03d}"

            f.write(f"{index}\n")
            f.write(f"{fmt_time(start)} --> {fmt_time(end)}\n")
            f.write(f"{text}\n\n")
            index += 1

def process_audio_and_generate_srt(file, language_code: str) -> str:
    """Full pipeline: save, convert, upload, transcribe, format .srt."""
    temp_dir = tempfile.mkdtemp()
    mp3_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp3")
    wav_path = mp3_path.replace(".mp3", ".wav")
    srt_path = mp3_path.replace(".mp3", ".srt")

    
    with open(mp3_path, "wb") as f_out:
        shutil.copyfileobj(file, f_out)

    
    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")

    
    blob_name = os.path.basename(wav_path)
    gcs_uri = upload_to_gcs(wav_path, blob_name)

    
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
    )

    print("Starting long_running_recognize...")
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=1800)
    print("Transcription complete")

    
    format_srt(response, srt_path)
    return srt_path
