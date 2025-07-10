import openai
import tempfile
import time
from typing import Tuple
import os
from dotenv import load_dotenv
load_dotenv(override=True)
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)  

def transcribe_with_gpt_4o_mini(file_bytes: bytes, language_code: str = "en") -> Tuple[str, float]:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
        temp_audio.write(file_bytes)
        temp_audio.flush()

        start_time = time.time()
        with open(temp_audio.name, "rb") as audio_file:
            transcript_response = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file,
                language=language_code,
                response_format="text"
            )
        end_time = time.time()

    latency = end_time - start_time
    return transcript_response, latency
