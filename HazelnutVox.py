import speech_recognition as sr
import numpy as np
import scipy.io.wavfile as wav
import simpleaudio as sa
import matplotlib.pyplot as plt
import time
from STTService import STTService
from TTSService import TTSService
from LLMService import LLMService

def play_audio(file_path):
    try:
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Nie udało się odtworzyć pliku: {e}")

def load_prompt(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return file.read().strip()
        except FileNotFoundError:
            print(f"Błąd: Nie znaleziono pliku {filepath}!")

def compare_spectrograms(human_audio, ai_audio_path, sample_rate_human=16000, filename="porownanie.png"):
    
    # 1. Wczytanie audio wygenerowanego przez AI
    ai_sample_rate, ai_audio_raw = wav.read(ai_audio_path)
    
    # Normalizacja sygnału AI do zakresu [-1.0, 1.0] 
    if ai_audio_raw.dtype == np.int16:
        ai_audio = ai_audio_raw.astype(np.float32) / 32768.0
    else:
        ai_audio = ai_audio_raw

    # Ustawienie dużej kanwy dla 4 wykresów
    plt.figure(figsize=(14, 10))
    
    # --- KOLUMNA 1: TWÓJ GŁOS ---
    plt.subplot(2, 2, 1)
    czas_h = np.linspace(0, len(human_audio) / sample_rate_human, num=len(human_audio))
    plt.plot(czas_h, human_audio, color='#1f77b4')
    plt.title("Człowiek - Przebieg czasowy")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.specgram(human_audio, Fs=sample_rate_human, cmap='inferno')
    plt.title("Człowiek - Spektrogram")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.colorbar(label="Intensywność [dB]")

    # --- KOLUMNA 2: GŁOS AI ---
    plt.subplot(2, 2, 2)
    czas_ai = np.linspace(0, len(ai_audio) / ai_sample_rate, num=len(ai_audio))
    plt.plot(czas_ai, ai_audio, color='#ff7f0e')
    plt.title("Sztuczna Inteligencja - Przebieg czasowy")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.specgram(ai_audio, Fs=ai_sample_rate, cmap='inferno')
    plt.title("Sztuczna Inteligencja - Spektrogram")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.colorbar(label="Intensywność [dB]")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    systemPrompt = load_prompt("prompt.txt")
    stt = STTService(model_size="turbo")
    tts = TTSService()
    llm = LLMService(model="llama3.2:1b")

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5 
    recognizer.non_speaking_duration = 0.5 
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 600

    print("\n--- System gotowy do działania ---")
    
    with sr.Microphone(sample_rate=16000) as source:
        print("Dostosowywanie do szumu tła (proszę o ciszę)...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        
        try:
            while True:
                print("\nMów teraz! (Naciśnij Ctrl+C, aby zatrzymać)")
                audio = recognizer.listen(source)
                start_time = time.time()
                audio_data = np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0
                
                tekst = stt.transcribe(audio_data)
                
                if tekst:
                    print(f"\n[STT] Rozpoznano: {tekst}")
                    response = llm.think(tekst,systemPrompt)
                    if response:
                        print(f"\n[LLM] Odpowiedziano: {response}")
                        
                        plik_wyjsciowy = "output.wav"
                        plik_audio = tts.generate_audio(response, plik_wyjsciowy)
                        
                        if plik_audio:
                            print("[Wizualizacja] Generowanie raportu AI vs Człowiek do porownanie.png...")
                            compare_spectrograms(human_audio=audio_data, ai_audio_path=plik_audio)
                            end_time = time.time()
                            processing_time = end_time - start_time
                            print(f"Czas przetwarzania: {processing_time:.2f} s\n")
                            play_audio(plik_audio)
                    
        except KeyboardInterrupt:
            print("\nZakończono działanie systemu.")

if __name__ == "__main__":
    main()