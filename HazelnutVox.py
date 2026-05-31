import speech_recognition as sr
import numpy as np
import scipy.io.wavfile as wav
import simpleaudio as sa
import matplotlib.pyplot as plt
import time
import asyncio
from STTService import STTService
from TTSService import TTSService
from LLMService import LLMService
import dataDrawer

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

async def main():
    systemPrompt = load_prompt("prompt.txt")
    stt = STTService(model_size="turbo")
    tts = TTSService(speaker_filename="VoiceJohny.wav")
    llm = LLMService(model="SpeakLeash/bielik-minitron-7B-v3.0-instruct:Q8_0", hostAddress="100.106.60.55")

    await llm.initialize()

    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 1.5 
    recognizer.non_speaking_duration = 1.0 
    recognizer.dynamic_energy_threshold = True
    
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
                    response = await llm.think(tekst, systemPrompt) 
                    if response:
                        print(f"\n[LLM] Odpowiedziano: {response}")
                        
                        plik_wyjsciowy = "output.wav"
                        plik_audio = tts.generate_audio(response, plik_wyjsciowy)
                        
                        if plik_audio:
                            dataDrawer.compare_spectrograms(human_audio=audio_data, ai_audio_path=plik_audio)
                            end_time = time.time()
                            processing_time = end_time - start_time
                            print(f"Czas przetwarzania: {processing_time:.2f} s\n")
                            play_audio(plik_audio)
                    
        except KeyboardInterrupt:
            print("\nZakończono działanie systemu.")

if __name__ == "__main__":
    asyncio.run(main())