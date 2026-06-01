# Hazelnut-Vox 🌰

## Overview
Hazelnut-Vox is a local, interactive voice agent implementing a complete STT-LLM-TTS (Speech-to-Text -> Large Language Model -> Text-to-Speech) pipeline. Originally developed as a module for a bot named "Orzech", it has evolved into a standalone project for the "Artificial Intelligence in Physical Signal Processing" academic course. 

The system actively listens to user input, processes the speech into text, generates a contextual response, and synthesizes it back into audio. It also includes built-in tools for physical signal analysis, generating visual comparisons of human and AI audio spectrograms.

## Architecture
The pipeline is built on modular Python services:
* **STT (Speech-to-Text):** Powered by OpenAI's Whisper (using the `turbo` model) to process incoming audio streams into text. It handles audio padding/trimming and log-Mel spectrogram generation for language detection and decoding.
* **LLM (Large Language Model):** Powered by Ollama running locally (default: `llama3.2:1b`). It includes a custom parser to strip `<think>` reasoning tags, ensuring clean textual output for the TTS module.
* **TTS (Text-to-Speech):** Utilizes XTTS v2 to generate natural-sounding voice responses.

## Key Features
* **Real-time Voice Interaction:** Automatically adjusts to ambient noise and listens dynamically using the `speech_recognition` library with custom energy thresholds.
* **Signal Analysis & Visualization:** Automatically generates a comparative visual report showing time waveforms and spectrograms (using `matplotlib` and `scipy`) of both the user's voice and the AI's synthesized response.
* **Fully Local Execution:** Designed to run entirely on local hardware, supporting CUDA acceleration for both Whisper and TTS models.

## Prerequisites
To run this project, you need:
* Python 3.8+
* CUDA Toolkit (highly recommended for GPU acceleration)
* Ollama installed and running locally.
* FFmpeg (required by Whisper)

## Environment Setup (Virtual Environment)

It is highly recommended to run this project inside an isolated virtual environment to keep dependencies organized and prevent conflicts with global system packages.

1. **Create the virtual environment:**
   Open a terminal in the root directory of the project and run:
   ```bash
   python -m venv .venv

2. **Activate the environment:**
   Depending on your operating system, activate the virtual environment using the appropriate command:
   ```bash
   .venv\Scripts\activate
   or
   source .venv/bin/activate

3. **Install dependencies:**
   While the virtual environment is active, install all the required packages from the configuration file:
   ```bash
   pip install -r requirements.txt