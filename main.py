from fastapi import FastAPI, File, UploadFile, Form
from transcribers.google_transcriber import transcribe_google
from transcribers.azure_transcriber import transcribe_azure_fast
from transcribers.azure_multilingual import transcribe_azure_fast_multilingual
from transcribers.aws_transcriber import transcribe_aws
from translators.LLMS.openai_translator import translate_text as translate_openai
from translators.LLMS.gemini_translator import translate_text_gemini
from text_to_speech.tts_google import tts_google
from text_to_speech.tts_aws import tts_aws
from text_to_speech.tts_azure import tts_azure
from fastapi.responses import FileResponse
from translators.services.azure_translate import translate_text_azure
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transcribers.google_new import transcribe_streaming_google
from transcribers.subtitle import process_audio_and_generate_outputs
import tempfile
from fastapi import Body 
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
    provider = provider.lower()
    start_time = time.time()
    file_ext = os.path.splitext(file.filename)[1].lower().lstrip(".")

    if file_ext not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="Only MP3 or WAV files are supported.")

    audio_data = await file.read()

    try:
        if provider == "google-new":
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name

            text = transcribe_streaming_google(temp_audio_path, language_code)

        elif provider == "google":
            text=transcribe_google(audio_data,language_code)
            

        elif provider == "azure-fast":
            text = transcribe_azure_fast(audio_data, language_code, file_type=file_ext)

        elif provider == "aws":
            text = transcribe_aws(audio_data, language_code)

        elif provider == "azure-fast-multilingual":
            text = transcribe_azure_fast_multilingual(audio_data)

        else:
            raise HTTPException(status_code=400, detail=f"Invalid transcription provider: {provider}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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


@app.post("/translate_service/azure")
def azure_translate(
    text: str = Form(...),
    target_language: str = Form(...),
    source_language: str | None = Form(None)
):
    try:
        start_time = time.time()
        translated_text = translate_text_azure(
            text,
            to_lang=target_language,
            from_lang=source_language
        )
        latency = time.time() - start_time

        return {
            "translated_text": translated_text,
            "latency": latency
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
from fastapi import FastAPI, UploadFile, File, Query


@app.post("/transcribed/subtitle_file")
async def subtitle_transcription(
    file: UploadFile = File(...),
    language_code: str = Form(...)
):
    zip_path = process_audio_and_generate_outputs(file.file, language_code)
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename="transcription_outputs.zip"
    )

from transcribers.openai_subtitlle import transcribe_and_diarize


import shutil

@app.post("/transcribe_whisper")
async def upload_and_transcribe(
    file: UploadFile = File(...),
    language_code: str = Form(...)
):
    input_path = f"temp_{file.filename}"
    with open(input_path, "wb") as f:
        f.write(await file.read())

    try:
        # Pass the language_code to the transcription function
        srt_file_path = transcribe_and_diarize(input_path, language_code=language_code)
        return FileResponse(
            path=srt_file_path,
            media_type="application/x-subrip",
            filename=os.path.basename(srt_file_path)
        )
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)