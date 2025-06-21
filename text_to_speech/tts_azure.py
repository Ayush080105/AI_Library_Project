import azure.cognitiveservices.speech as speechsdk
import uuid
import os

def tts_azure(text: str, language_code: str) -> str:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_REGION")

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    speech_config.speech_synthesis_voice_name = language_code

    
    filename = f"azure_tts_{uuid.uuid4().hex}.mp3"
    audio_output = speechsdk.audio.AudioOutputConfig(filename=filename)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_output)
    result = synthesizer.speak_text_async(text).get()

    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise Exception(f"Speech synthesis failed: {result.reason}")

    return filename  
