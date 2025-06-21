import azure.cognitiveservices.speech as speechsdk
from azure.cognitiveservices.speech import CancellationDetails
import tempfile
import os
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

        
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_REGION)
        speech_config.speech_recognition_language = language_code
        audio_config = speechsdk.audio.AudioConfig(filename=temp_audio_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = speechsdk.CancellationDetails(result)
            return f"Recognition canceled: {cancellation.reason}, ErrorDetails: {cancellation.error_details}"
        else:
            return f"Recognition failed: {result.reason}"

    except Exception as e:
        return f"Error during transcription: {str(e)}"

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass
