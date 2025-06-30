import os
import requests
from dotenv import load_dotenv

load_dotenv()

AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")

def translate_text_azure(text: str, to_lang: str, from_lang: str = None) -> str:
    if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_REGION:
        raise Exception("Azure credentials not set in .env")

    url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate?api-version=3.0&to={to_lang}"
    if from_lang:
        url += f"&from={from_lang}"

    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_TRANSLATOR_KEY,
        "Ocp-Apim-Subscription-Region": AZURE_TRANSLATOR_REGION,
        "Content-type": "application/json"
    }

    body = [{"text": text}]
    response = requests.post(url, headers=headers, json=body)

    if response.status_code != 200:
        raise Exception(f"Azure Translation failed: {response.status_code} - {response.text}")

    return response.json()[0]["translations"][0]["text"]
