import boto3
import uuid
import os

def tts_aws(text: str, language_code: str) -> str:
    polly = boto3.client("polly")
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Joanna" if language_code.startswith("en") else "Aditi"  
    )

    output_dir = "tts_output"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{output_dir}/aws_tts_{uuid.uuid4().hex}.mp3"
    with open(filename, "wb") as f:
        f.write(response["AudioStream"].read())

    return filename
