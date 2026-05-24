import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import librosa
import librosa.display

def compare_spectrograms(human_audio, ai_audio_path, sample_rate_human=16000, filename="porownanie.png"):
    
    # Wczytanie audio wygenerowanego przez AI
    ai_sample_rate, ai_audio_raw = wav.read(ai_audio_path)
    
    # Normalizacja sygnału AI
    if ai_audio_raw.dtype == np.int16:
        ai_audio = ai_audio_raw.astype(np.float32) / 32768.0
    else:
        ai_audio = ai_audio_raw.astype(np.float32)

    # Bezpieczna normalizacja sygnału człowieka
    if human_audio.dtype == np.int16:
        human_audio_float = human_audio.astype(np.float32) / 32768.0
    else:
        human_audio_float = human_audio.astype(np.float32)

    plt.figure(figsize=(14, 20))
    
    # ==========================================
    # --- KOLUMNA 1: GŁOS LUDZKI ---
    # ==========================================
    
    # 1. Przebieg sygnału w czasie
    plt.subplot(4, 2, 1)
    czas_h = np.linspace(0, len(human_audio) / sample_rate_human, num=len(human_audio))
    plt.plot(czas_h, human_audio, color='#1f77b4')
    plt.title("Człowiek - Przebieg czasowy")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True, alpha=0.3)

    # 2. FFT (Widmo amplitudowe)
    plt.subplot(4, 2, 3)
    yf_h = np.fft.rfft(human_audio)
    xf_h = np.fft.rfftfreq(len(human_audio), 1 / sample_rate_human)
    amp_h = np.abs(yf_h)
    
    plt.plot(xf_h, amp_h, color='#1f77b4')
    plt.title("Człowiek - Widmo amplitudowe (FFT)")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.grid(True, alpha=0.3)

    # 3. Spektrogram liniowy
    plt.subplot(4, 2, 5)
    plt.specgram(human_audio, Fs=sample_rate_human, cmap='inferno')
    plt.title("Człowiek - Spektrogram liniowy")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.colorbar(label="Intensywność [dB]")

    # 4. Mel-spektrogram
    plt.subplot(4, 2, 7)
    mel_h = librosa.feature.melspectrogram(y=human_audio_float, sr=sample_rate_human)
    mel_h_db = librosa.power_to_db(mel_h, ref=np.max) # Konwersja mocy na decybele
    librosa.display.specshow(mel_h_db, sr=sample_rate_human, x_axis='time', y_axis='mel', cmap='inferno')
    plt.title("Człowiek - Mel-spektrogram")
    plt.colorbar(format='%+2.0f dB')


    # ==========================================
    # --- KOLUMNA 2: GŁOS AI ---
    # ==========================================
    
    # 1. Przebieg sygnału w czasie
    plt.subplot(4, 2, 2)
    czas_ai = np.linspace(0, len(ai_audio) / ai_sample_rate, num=len(ai_audio))
    plt.plot(czas_ai, ai_audio, color='#ff7f0e')
    plt.title("Sztuczna Inteligencja - Przebieg czasowy")
    plt.xlabel("Czas [s]")
    plt.ylabel("Amplituda")
    plt.grid(True, alpha=0.3)

    # 2. FFT (Widmo amplitudowe)
    plt.subplot(4, 2, 4)
    yf_ai = np.fft.rfft(ai_audio)
    xf_ai = np.fft.rfftfreq(len(ai_audio), 1 / ai_sample_rate)
    amp_ai = np.abs(yf_ai)
    
    plt.plot(xf_ai, amp_ai, color='#ff7f0e')
    plt.title("Sztuczna Inteligencja - Widmo amplitudowe (FFT)")
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda")
    plt.grid(True, alpha=0.3)

    # 3. Spektrogram liniowy
    plt.subplot(4, 2, 6)
    plt.specgram(ai_audio, Fs=ai_sample_rate, cmap='inferno')
    plt.title("Sztuczna Inteligencja - Spektrogram liniowy")
    plt.xlabel("Czas [s]")
    plt.ylabel("Częstotliwość [Hz]")
    plt.colorbar(label="Intensywność [dB]")
    
    # 4. Mel-spektrogram
    plt.subplot(4, 2, 8)
    mel_ai = librosa.feature.melspectrogram(y=ai_audio, sr=ai_sample_rate)
    mel_ai_db = librosa.power_to_db(mel_ai, ref=np.max)
    librosa.display.specshow(mel_ai_db, sr=ai_sample_rate, x_axis='time', y_axis='mel', cmap='inferno')
    plt.title("Sztuczna Inteligencja - Mel-spektrogram")
    plt.colorbar(format='%+2.0f dB')
    
    # Finalizacja
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()