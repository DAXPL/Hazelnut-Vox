import speech_recognition as sr
import numpy as np
import scipy.io.wavfile as wav
import simpleaudio as sa
import matplotlib.pyplot as plt
import time
import asyncio
import torch
import re
from STTService import STTService
from TTSService import TTSService
from LLMService import LLMService
import dataDrawer
import numpy as np
import scipy.io.wavfile as wav

def trim_silence(filepath, threshold=0.01, margin_samples=1000):
    try:
        sample_rate, data = wav.read(filepath)
        if len(data.shape) > 1:
            mono_data = np.mean(data, axis=1)
        else:
            mono_data = data
        max_amplitude = np.max(np.abs(mono_data))
        if max_amplitude == 0:
            return filepath
        silence_threshold = max_amplitude * threshold
        non_silent_indices = np.where(np.abs(mono_data) > silence_threshold)[0]
        
        if len(non_silent_indices) == 0:
            return filepath
        start_index = max(0, non_silent_indices[0] - margin_samples)
        end_index = min(len(data), non_silent_indices[-1] + margin_samples)
        trimmed_data = data[start_index:end_index]
        wav.write(filepath, sample_rate, trimmed_data)
        
        return filepath
    except Exception as e:
        print(f"Błąd podczas przycinania ciszy: {e}")
        return filepath

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

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]


async def tts_worker(tts, text_queue, audio_queue):
    while True:
        item = await text_queue.get()
        if item is None: 
            await audio_queue.put(None)
            text_queue.task_done()
            break
            
        text, idx = item
        output_path = f"output_part_{idx}.wav"
        plik_audio = await asyncio.to_thread(tts.generate_audio, text, output_path)
        
        if plik_audio:
            plik_audio = await asyncio.to_thread(trim_silence, plik_audio)
            await audio_queue.put(plik_audio)
            
        text_queue.task_done()

async def audio_player_worker(audio_queue):
    while True:
        file_path = await audio_queue.get()
        if file_path is None:
            audio_queue.task_done()
            break
            
        await asyncio.to_thread(play_audio, file_path)
        audio_queue.task_done()


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
                
                tekst = stt.transcribe(audio_data, transcribeLanguage="pl")
                
                if tekst:
                    print(f"\n[STT] Rozpoznano: {tekst}")
                    response = await llm.think(tekst, systemPrompt) 
                    
                    if response:
                        print(f"\n[LLM] Odpowiedziano: {response}")
                        torch.cuda.empty_cache()
                        
                        zdania = split_into_sentences(response)
                        
                        text_queue = asyncio.Queue()
                        audio_queue = asyncio.Queue()
                        
                        task_tts = asyncio.create_task(tts_worker(tts, text_queue, audio_queue))
                        task_player = asyncio.create_task(audio_player_worker(audio_queue))
                        
                        # Wrzucamy pocięte zdania do kolejki tekstowej
                        for i, zdanie in enumerate(zdania):
                            await text_queue.put((zdanie, i))
                            
                        end_time = time.time()
                        print(f"Całkowity czas iteracji: {(end_time - start_time):.2f} s\n")

                        # (None) żeby workery wiedziały, kiedy skończyć
                        await text_queue.put(None)
                        
                        await text_queue.join()
                        await audio_queue.join()
                        
                        
                    
        except KeyboardInterrupt:
            print("\nZakończono działanie systemu.")

if __name__ == "__main__":
    asyncio.run(main())