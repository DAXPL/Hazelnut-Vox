import os
import torch
from torch import autocast
from TTS.api import TTS
import transformers
transformers.logging.set_verbosity_error()

original_load = torch.load
def unsafe_load(args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(args, **kwargs)
torch.load = unsafe_load

class TTSService:
    def __init__(self, speaker_filename="Voice.wav"):
        print("Ładowanie modelu TTS ...")
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_models_path = os.path.join(self.app_dir, "models")
        
        if not os.path.exists(self.local_models_path):
            os.makedirs(self.local_models_path)
            
        os.environ["TTS_HOME"] = self.local_models_path
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Używane urządzenie: {self.device}")
        print("Ładowanie modelu ...")
        
        self.speaker_wav = os.path.join(self.app_dir, speaker_filename)
        if not os.path.exists(self.speaker_wav):
            print(f"OSTRZEŻENIE: Nie znaleziono pliku referencyjnego głosu: {self.speaker_wav}!")
                
        print("Ładowanie wag modelu ...")
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        print("Model gotowy.")

    def generate_audio(self, text_content, output_path="file.wav", audioLanguage="pl"):
        if not text_content:
            return None
            
        try:
            self.tts.tts_to_file(
                text=text_content,
                speaker_wav=self.speaker_wav,
                language=audioLanguage,
                file_path=output_path,
                split_sentences=False,
                temperature=0.75,
                top_p=0.85,
                top_k=50,
                speed=1.25
            )
            return output_path
        except Exception as e:
            print(f"Błąd podczas generowania: {e}")
            return None