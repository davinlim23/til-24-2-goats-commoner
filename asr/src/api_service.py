from fastapi import FastAPI, Request
import base64
from ASRManager import ASRManager
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration,WhisperProcessor
app = FastAPI()
new_model = WhisperForConditionalGeneration.from_pretrained('models/best_ASR_model')
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
asr_manager = ASRManager(model=new_model,processor=processor,device = device)


@app.get("/health")
def health():
    return {"message": "health ok"}


@app.post("/stt")
async def stt(request: Request):
    """
    Performs ASR given the filepath of an audio file
    Returns transcription of the audio
    """

    # get base64 encoded string of audio, convert back into bytes
    input_json = await request.json()

    predictions = []
    for instance in input_json["instances"]:
        # each is a dict with one key "b64" and the value as a b64 encoded string
        audio_bytes = base64.b64decode(instance["b64"])
        transcription = asr_manager.transcribe(audio_bytes)
        # print(type(transcription))
        predictions.append(transcription)

    return {"predictions": predictions}
