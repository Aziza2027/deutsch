import torch
from TTS.api import TTS

class TextToSpeech:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Init TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
    
    def to_audio_en(self, text, path):
        # Generate audio in English
        self.tts.tts_to_file(text=text, speaker_wav=path, language="en", file_path="output.wav")
        
    def to_audio_de(self, text, path):
        # Generate audio in German
        self.tts.tts_to_file(text=text, speaker_wav=path, language="de", file_path="output_de.wav")


