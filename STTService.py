from faster_whisper import WhisperModel
import torch
import numpy as np

# https://github.com/SYSTRAN/faster-whisper
class STTService:
    def __init__(self, model_size="turbo"):
        print(f"Ładowanie modelu STT (faster-whisper: {model_size})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "int8_float16" if self.device == "cuda" else "float32"
        print(f'STT Device: {self.device} (compute_type: {self.compute_type})')
        self.model = WhisperModel(model_size, device=self.device, compute_type=self.compute_type)
        print("Model STT gotowy.")

    def transcribe(self, audio_data, transcribeLanguage="dynamic"):
        if isinstance(audio_data, np.ndarray):
            audio = audio_data.astype(np.float32)
        else:
            audio = audio_data

        target_language = None if transcribeLanguage == "dynamic" else transcribeLanguage

        try:
            segments, info = self.model.transcribe(
                audio, 
                language=target_language,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            if transcribeLanguage == "dynamic":
                print(f"Wykryty język: {info.language} (prawdopodobieństwo: {info.language_probability:.2f})")

            tekst = ""
            for segment in segments:
                tekst += segment.text + " "

            return tekst.strip()
            
        except Exception as e:
            print(f"Błąd podczas transkrypcji STT: {e}")
            return ""