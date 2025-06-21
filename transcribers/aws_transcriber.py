import boto3
import uuid
import time
import os
import tempfile
from dotenv import load_dotenv

load_dotenv(override=True)  

def transcribe_aws(audio_bytes: bytes, language_code: str) -> str:
    region_name = os.getenv("AWS_REGION")
    bucket = os.getenv("AWS_BUCKET_NAME")
    if not region_name:
        raise Exception("AWS_REGION is not set in .env")
    if not bucket:
        raise Exception("AWS_BUCKET_NAME is not set in .env")

  
    s3 = boto3.client('s3', region_name=region_name)
    transcribe = boto3.client('transcribe', region_name=region_name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name
        object_key = f"audio/{uuid.uuid4()}.wav"
        s3.upload_file(temp_audio_path, bucket, object_key)

    job_name = f"job-{uuid.uuid4()}"
    job_uri = f"s3://{bucket}/{object_key}"

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': job_uri},
        MediaFormat='wav',
        LanguageCode=language_code
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
        time.sleep(5)

    if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
        transcript_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
        import requests
        result = requests.get(transcript_url).json()
        return result['results']['transcripts'][0]['transcript']
    else:
        return "Transcription failed"
