import os
import json
import time
import tempfile
import mimetypes
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")


def transcribe_azure_fast(audio_bytes: bytes, language_code: str = "en-US", file_type: str = "wav") -> str:
    temp_audio_path = None
    try:
        
        extension = ".mp3" if file_type.lower() == "mp3" else ".wav"
        mime_type = "audio/mpeg" if file_type.lower() == "mp3" else "audio/wav"

        
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension, mode='wb') as f:
            temp_audio_path = f.name
            f.write(audio_bytes)

        
        url = f"https://{AZURE_REGION}.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2024-11-15"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Accept": "application/json"
        }

        
        with open(temp_audio_path, "rb") as audio_file:
            files = [
                ('audio', (f.name, audio_file, mime_type)),
                ('definition', (None, json.dumps({
                    "locales": [language_code],
                    "profanityFilterMode": "Masked",
                    "diarizationSettings": {"minSpeakers": 1, "maxSpeakers": 2},
                    "channels": [0]
                }), 'application/json'))
            ]

            response = requests.post(url, headers=headers, files=files)

        
        if response.status_code != 200:
            return f"Azure Fast Transcription failed: {response.status_code} - {response.text}"

        job = response.json()

        
        if "combinedPhrases" in job:
            return " ".join([p["text"] for p in job.get("combinedPhrases", [])]) or "Transcription completed but no text found."

        
        if "id" not in job:
            return f"Azure Fast Transcription failed: No 'id' in response: {job}"

        job_id = job["id"]
        poll_url = f"https://{AZURE_REGION}.api.cognitive.microsoft.com/speechtotext/transcriptions/{job_id}?api-version=2024-11-15"

        while True:
            poll_resp = requests.get(poll_url, headers=headers)
            if poll_resp.status_code != 200:
                return f"Polling failed: {poll_resp.status_code} - {poll_resp.text}"

            poll_data = poll_resp.json()
            status = poll_data.get("status", "")
            if status == "Succeeded":
                break
            elif status == "Failed":
                return f"Transcription job failed: {poll_data}"
            time.sleep(5)

        files_url = poll_data["links"]["files"]
        files_resp = requests.get(files_url, headers=headers)
        if files_resp.status_code != 200:
            return f"Failed to fetch transcription files: {files_resp.status_code} - {files_resp.text}"

        files_data = files_resp.json()
        transcript_file_url = next(
            (f["links"]["contentUrl"] for f in files_data["values"] if f["kind"] == "Transcription"), None
        )

        if not transcript_file_url:
            return "Transcription succeeded, but no result file found."

        transcript_data = requests.get(transcript_file_url).json()
        text = " ".join([p["display"] for p in transcript_data.get("recognizedPhrases", [])])
        return text or "Transcription was successful but no text found."

    except Exception as e:
        return f"Error during Azure Fast Transcription: {str(e)}"

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
