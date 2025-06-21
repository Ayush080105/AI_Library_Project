from fastapi import FastAPI, File, UploadFile, Form
from transcribers.google_transcriber import transcribe_google
from transcribers.azure_transcriber import transcribe_azure
from transcribers.aws_transcriber import transcribe_aws
from translators.LLMS.openai_translator import translate_text as translate_openai
from translators.LLMS.gemini_translator import translate_text_gemini
from text_to_speech.tts_google import tts_google
from text_to_speech.tts_aws import tts_aws
from text_to_speech.tts_azure import tts_azure
from fastapi.responses import FileResponse
import io
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.post("/transcribe/{provider}")
async def transcribe(
    provider: str,
    language_code: str = Form(...),
    file: UploadFile = File(...)
):
    audio_data = await file.read()
    start_time = time.time()

    provider = provider.lower()
    if provider == "google":
        text = transcribe_google(audio_data, language_code)
    elif provider == "azure":
        text = transcribe_azure(audio_data, language_code)
    elif provider == "aws":
        text = transcribe_aws(audio_data, language_code)
    else:
        return {"error": f"Invalid transcription provider: {provider}"}

    latency = time.time() - start_time
    return {"transcription": text, "latency": latency}


@app.post("/translate/{provider}")
async def translate(
    provider: str,
    text: str = Form(...),
    target_language: str = Form(...)
):
    start_time = time.time()

    provider = provider.lower()
    try:
        if provider == "openai":
            translated = translate_openai(text, target_language)
        elif provider == "gemini":
            translated = translate_text_gemini(text, target_language)
        else:
            return {"error": f"Unsupported translation provider: {provider}"}

        latency = time.time() - start_time
        return {"translated_text": translated, "latency": latency}

    except Exception as e:
        return {"error": str(e)}


@app.post("/tts/{provider}")
async def tts(
    provider: str,
    text: str = Form(...),
    language_code: str = Form(...)
):
    start_time = time.time()
    provider = provider.lower()

    try:
        if provider == "google":
            file_path = tts_google(text, language_code)
        elif provider == "aws":
            file_path = tts_aws(text, language_code)
        elif provider == "azure":
            file_path = tts_azure(text, language_code)
        else:
            return {"error": f"Unsupported TTS provider: {provider}"}

        latency = time.time() - start_time
        return FileResponse(
            path=file_path,
            media_type="audio/mpeg",
            filename=os.path.basename(file_path),
            headers={"X-Generation-Latency": str(latency)}
        )

    except Exception as e:
        return {"error": f"TTS failed: {str(e)}"}
