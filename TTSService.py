import os
import torch
from TTS.api import TTS

class TTSService:
    def __init__(self):
        print("Ładowanie modelu TTS ...")
        self.app_dir = os.path.dirname(os.path.abspath(__file__))
        self.local_models_path = os.path.join(self.app_dir, "models")
        
        if not os.path.exists(self.local_models_path):
            os.makedirs(self.local_models_path)
            
        os.environ["TTS_HOME"] = self.local_models_path
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Używane urządzenie: {self.device}")
        print("Ładowanie modelu ...")
        
        self.tts = TTS(model_name="tts_models/pl/mai_female/vits", progress_bar=False).to(self.device)
        print("Model gotowy.")

    def generate_audio(self, text_content, output_path="file.wav"):
        if not text_content:
            return None
            
        try:
            print(f"Przetwarzanie: '{text_content}' - generowanie audio...")
            self.tts.tts_to_file(text=text_content, file_path=output_path)
            return output_path
        except Exception as e:
            print(f"Błąd podczas generowania: {e}")
            return None