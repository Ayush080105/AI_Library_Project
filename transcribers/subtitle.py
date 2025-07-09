import os
import uuid
import tempfile
import shutil
import datetime
import zipfile
from pydub import AudioSegment
from google.cloud import storage
from google.cloud import speech_v1p1beta1 as speech

BUCKET_NAME = "ayush_bucket_0716"  

def upload_to_gcs(local_path: str, dest_blob_name: str) -> str:
    client = storage.Client()
    bucket = storage.Client().bucket(BUCKET_NAME)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{BUCKET_NAME}/{dest_blob_name}"

def format_srt(response, output_path: str):
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

def format_speaker_transcript(response, output_path: str):
    result = response.results[-1]
    words = result.alternatives[0].words

    with open(output_path, "w", encoding="utf-8") as f:
        current_speaker = None
        line = ""

        for word in words:
            speaker = word.speaker_tag
            if speaker != current_speaker:
                if line:
                    f.write(f"Speaker {current_speaker}: {line.strip()}\n")
                line = word.word + " "
                current_speaker = speaker
            else:
                line += word.word + " "
        if line:
            f.write(f"Speaker {current_speaker}: {line.strip()}\n")

def process_audio_and_generate_outputs(file, language_code: str) -> str:
    temp_dir = tempfile.mkdtemp()
    mp3_path = os.path.join(temp_dir, f"{uuid.uuid4()}.mp3")
    wav_path = mp3_path.replace(".mp3", ".wav")
    srt_path = mp3_path.replace(".mp3", ".srt")
    txt_path = mp3_path.replace(".mp3", "_speakers.txt")
    zip_path = mp3_path.replace(".mp3", ".zip")

    with open(mp3_path, "wb") as f_out:
        shutil.copyfileobj(file, f_out)

    audio = AudioSegment.from_file(mp3_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(wav_path, format="wav")

    gcs_uri = upload_to_gcs(wav_path, os.path.basename(wav_path))

    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code=language_code,
        enable_word_time_offsets=True,
        enable_automatic_punctuation=True,
        enable_speaker_diarization=True,
        diarization_speaker_count=2,
    )

    print("Starting transcription with speaker diarization...")
    operation = client.long_running_recognize(config=config, audio=audio)
    response = operation.result(timeout=1800)
    print("Transcription complete")

    format_srt(response, srt_path)
    format_speaker_transcript(response, txt_path)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(srt_path, arcname="transcript.srt")
        zipf.write(txt_path, arcname="transcript_speakers.txt")

    return zip_path
