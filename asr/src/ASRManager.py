import torchaudio
import soundfile as sf
import numpy as np
import io
import noisereduce as nr


def denoise_data(audio,rate):
    # Perform noise reduction
    noisy_part = audio[0:int(rate*0.5)]  # Identify the noisy part
    reduced_noise_audio = nr.reduce_noise(y=audio, sr=rate, y_noise=noisy_part)
    return reduced_noise_audio

def prepare_for_inference(audio_path,processor):
   # load audio data
    waveform, sample_rate = sf.read(io.BytesIO(audio_path), dtype='float32')
    # print(waveform.shape)
    
    waveform = denoise_data(waveform,sample_rate)

    # compute log-Mel input features from input audio array 
    input_features = processor(waveform, sampling_rate = sample_rate,return_tensors="pt").input_features
    
    return input_features

class ASRManager:
    def __init__(self,model,processor,device):
        # initialize the model here
        self.device = device
        self.model = model
        self.model.generation_config.forced_decoder_ids = None
        self.model.to(device)
        self.processor = processor

    def transcribe(self, audio_bytes: bytes) -> str:
        
        input_features = prepare_for_inference(audio_bytes,self.processor).to(self.device)
        
        prediction_ids = self.model.generate(input_features)

        transcription = self.processor.batch_decode(prediction_ids, skip_special_tokens=True)
        
        return transcription[0]
