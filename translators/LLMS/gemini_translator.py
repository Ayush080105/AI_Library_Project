import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash")

def translate_text_gemini(text: str, target_language: str) -> str:
    prompt = (
    f"Translate the following text to {target_language} only. "
    f"Do not provide any transliteration or explanation. "
    f"Just return the translated sentence in {target_language} script:\n\n"
    f"{text}"
)
    response = model.generate_content(prompt)
    return response.text.strip()
