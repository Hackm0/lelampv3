import sounddevice as sd
import numpy as np
import whisper
import subprocess
import warnings
import os
import google.generativeai as genai
from dotenv import load_dotenv
from scipy.io.wavfile import write
import queue
import threading
import time

# ----------------------------
# Setup TTS safely using espeak as fallback
# ----------------------------
tts = None

def init_pyttsx3():
    """Try to initialize pyttsx3 with a working voice"""
    global tts
    try:
        import pyttsx3
        tts = pyttsx3.init()
        tts.setProperty("rate", 165)
        # Don't try to set voices - use the default
        # Setting voices often fails on Linux with espeak backend
        return True
    except Exception as e:
        print(f"pyttsx3 initialization failed: {e}")
        return False

use_pyttsx3 = init_pyttsx3()

def speak(text):
    """Speak text aloud using espeak directly (most reliable on Linux)"""
    # Use espeak directly - it's the most reliable on Linux/Raspberry Pi
    try:
        subprocess.run(['espeak', '-s', '165', text], check=True, 
                      stderr=subprocess.DEVNULL)
        return
    except FileNotFoundError:
        pass
    
    # Try espeak-ng if espeak is not available
    try:
        subprocess.run(['espeak-ng', '-s', '165', text], check=True,
                      stderr=subprocess.DEVNULL)
        return
    except FileNotFoundError:
        pass
    
    # Last resort: try pyttsx3
    global tts, use_pyttsx3
    if use_pyttsx3 and tts:
        try:
            tts.say(text)
            tts.runAndWait()
            return
        except Exception as e:
            use_pyttsx3 = False
    
    print(f"No TTS available. Response: {text}")

# ----------------------------
# Play a sound
# ----------------------------
def play_sound(sound_path):
    """Plays a sound file."""
    try:
        # Use aplay for broader compatibility on Linux
        subprocess.run(['aplay', sound_path], check=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Could not play sound {sound_path}: {e}")

# ----------------------------
# VAD (Voice Activity Detection)
# ----------------------------
# A simple energy-based VAD
def is_speech(chunk, threshold=0.01):
    """Check if audio chunk contains speech based on RMS energy."""
    rms = np.sqrt(np.mean(chunk**2))
    return rms > threshold

# ----------------------------
# Find audio device (Seeed preferred, fallback to default)
# ----------------------------
def get_audio_device(output=True):
    """Find Seeed audio device or fall back to default system device"""
    devices = sd.query_devices()
    
    # First, try to find Seeed device
    seeed_devices = [(i, d) for i, d in enumerate(devices) if "seeed" in d['name'].lower()]
    for i, d in seeed_devices:
        if output and d['max_output_channels'] > 0:
            print(f"Using Seeed output device: {d['name']}")
            return i
        if not output and d['max_input_channels'] > 0:
            print(f"Using Seeed input device: {d['name']}")
            return i
    
    # Fallback to default system device
    try:
        if output:
            default = sd.default.device[1]  # output device
            if default is not None:
                print(f"Seeed not found, using default output device: {devices[default]['name']}")
                return default
        else:
            default = sd.default.device[0]  # input device
            if default is not None:
                print(f"Seeed not found, using default input device: {devices[default]['name']}")
                return default
    except:
        pass
    
    # Last resort: find any device with input/output channels
    for i, d in enumerate(devices):
        if output and d['max_output_channels'] > 0:
            print(f"Using fallback output device: {d['name']}")
            return i
        if not output and d['max_input_channels'] > 0:
            print(f"Using fallback input device: {d['name']}")
            return i
    
    return None

# List available devices for debugging
print("Available audio devices:")
for i, d in enumerate(sd.query_devices()):
    print(f"  {i}: {d['name']} (in: {d['max_input_channels']}, out: {d['max_output_channels']})")
print()

audio_output = get_audio_device(output=True)
audio_input = get_audio_device(output=False)

if audio_input is None:
    raise RuntimeError("No audio input device found!")
if audio_output is None:
    print("Warning: No audio output device found, TTS may not work")

# ----------------------------
# Main application loop
# ----------------------------
def main_loop():
    """The main application loop for the voice assistant."""
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
    print("Loading Whisper model for transcription...")
    transcribe_model = whisper.load_model("base", device="cpu")
    print("Whisper transcription model loaded.")

    sample_rate = 16000  # Whisper's preferred sample rate
    
    # --- VAD parameters ---
    silence_threshold = 0.01  # RMS threshold for silence
    silence_duration_ms = 1500 # Stop recording after 1.5s of silence
    chunk_duration_ms = 100 # Process audio in 100ms chunks
    chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
    silence_chunks = int(silence_duration_ms / chunk_duration_ms)
    
    while True:
        # Wake word detection has been removed. The loop now starts listening immediately.
        
        # --- 1. Record user's command with VAD ---
        print("\nListening for your command...")
        audio_chunks = []
        silent_chunks_count = 0
        
        # Use a stream to capture audio continuously
        with sd.InputStream(samplerate=sample_rate, channels=1, device=audio_input, blocksize=chunk_samples) as stream:
            while silent_chunks_count < silence_chunks:
                audio_chunk, overflowed = stream.read(chunk_samples)
                if overflowed:
                    print("Warning: Input overflowed!")
                
                audio_chunks.append(audio_chunk)
                
                if is_speech(audio_chunk, threshold=silence_threshold):
                    silent_chunks_count = 0
                else:
                    silent_chunks_count += 1
            
        print("Finished recording.")
        recording = np.concatenate(audio_chunks)
        
        # Save recording to a temporary file
        temp_audio_file = "temp_input.wav"
        write(temp_audio_file, sample_rate, recording)

        # --- 3. Transcribe the audio ---
        try:
            result = transcribe_model.transcribe(temp_audio_file, fp16=False)
            user_text = result["text"].strip()
            print("You said:", user_text)

            if not user_text:
                speak("I didn't catch that. Please try again.")
                continue

        except Exception as e:
            print(f"Transcription failed: {e}")
            speak("Sorry, I had trouble understanding you.")
            continue

        # --- 4. Generate response from Gemini ---
        try:
            prompt = f"""
            You are a gentle, reflective homework helper lamp. The student said: "{user_text}".
            Give a thoughtful, encouraging response. Suggest rethinking strategies or hints, 
            but do not give the full answer unless asked explicitly.
            """
            
            # API key is loaded from .env at the start
            genai_model = genai.GenerativeModel("gemini-1.5-flash")
            response = genai_model.generate_content(prompt)
            lamp_response = response.text.strip()

        except Exception as e:
            print(f"Gemini API call failed: {e}")
            speak("I'm having trouble connecting to my brain right now.")
            continue

        # --- 5. Speak the response ---
        print("LeLamp:", lamp_response)
        speak(lamp_response)


if __name__ == "__main__":
    # --- Setup API Key ---
    dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(dotenv_path):
        print("ERROR: .env file not found. Please create one with your GEMINI_API_KEY.")
    load_dotenv(dotenv_path)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "REPLACE_WITH_YOUR_API_KEY":
        raise ValueError("GEMINI_API_KEY not found or not set in .env file.")
    genai.configure(api_key=api_key)

    # --- Wake Word Detection has been removed ---

    # --- Start main application loop ---
    # The main loop now runs directly in the main thread.
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
