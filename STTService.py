import whisper
import torch
#https://github.com/openai/whisper
class STTService:
    def __init__(self, model_size="turbo"):
        print("Ładowanie modelu STT (Whisper)...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.useFP16 = True if torch.cuda.is_available() else False
        print('STT Device:', self.device)
        
        self.model = whisper.load_model(model_size, device=self.device)

    def transcribe(self, audio_data):
        import numpy as np

        # 1. audio_data jest już tablicą, upewniamy się tylko, że ma typ float32
        # Whisper wymaga jednowymiarowej tablicy (mono) z częstotliwością próbkowania 16000 Hz
        if isinstance(audio_data, np.ndarray):
            audio = audio_data.astype(np.float32)
        else:
            audio = audio_data

        # 2. Przycięcie lub dopełnienie (padding) sygnału do 30 sekund
        audio = whisper.pad_or_trim(audio)

        # 3. Tworzenie log-Mel spektrogramu i przeniesienie go na GPU/CPU
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.model.device)

        # 4. Wykrywanie języka na podstawie spektrogramu
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)
        print(f"Wykryty język: {detected_lang}")

        # 5. Dekodowanie audio
        # Podajemy wykryty język i flagę fp16, którą zdefiniowałeś w __init__
        options = whisper.DecodingOptions(language=detected_lang, fp16=self.useFP16)
        result = whisper.decode(self.model, mel, options)
        
        # Zwróć uwagę: funkcja decode() zwraca obiekt DecodingResult.
        # Odwołujemy się do tekstu przez atrybut .text, a nie klucz ["text"]
        return result.text.strip()