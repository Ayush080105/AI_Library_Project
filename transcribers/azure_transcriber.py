import os
import tempfile
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_REGION = os.getenv("AZURE_REGION")

def transcribe_azure(audio_bytes: bytes, language_code: str) -> str:
    temp_audio_path = None
    try:
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", mode='wb') as f:
            temp_audio_path = f.name
            f.write(audio_bytes)

        
        url = f"https://{AZURE_REGION}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
        params = {
            "language": language_code  
        }

        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_SPEECH_KEY,
            "Content-Type": "audio/wav"
        }

        with open(temp_audio_path, "rb") as audio_file:
            response = requests.post(url, params=params, headers=headers, data=audio_file)

        if response.status_code != 200:
            return f"Azure Fast Transcription failed: {response.status_code} - {response.text}"

        result = response.json()
        return result.get("DisplayText", "No transcription found")

    except Exception as e:
        return f"Error during transcription: {str(e)}"

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
