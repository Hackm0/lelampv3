import sounddevice as sd
import numpy as np
import pyttsx3
import whisper
import openai
from scipy.io.wavfile import write

# ----------------------------
# Setup TTS
# ----------------------------
tts = pyttsx3.init()
tts.setProperty("rate", 165)

# Ensure a valid voice is selected
voices = tts.getProperty('voices')
if voices:
    tts.setProperty('voice', voices[0].id)  # pick the first available voice

def speak(text):
    """Speak text aloud using pyttsx3"""
    tts.say(text)
    tts.runAndWait()

# ----------------------------
# Find Seeed audio device
# ----------------------------
def get_seeed_device(output=True):
    devices = sd.query_devices()
    seeed_devices = [(i, d) for i, d in enumerate(devices) if "seeed" in d['name'].lower()]
    for i, d in seeed_devices:
        if output and d['max_output_channels'] > 0:
            return i
        if not output and d['max_input_channels'] > 0:
            return i
    return None

seeed_output = get_seeed_device(output=True)
seeed_input  = get_seeed_device(output=False)

if seeed_output is None or seeed_input is None:
    raise RuntimeError("Seeed audio device not found!")

# ----------------------------
# Record audio
# ----------------------------
duration = 5  # seconds
sample_rate = 44100

print("Listening for 5 seconds...")
recording = sd.rec(
    int(duration * sample_rate),
    samplerate=sample_rate,
    channels=1,
    device=seeed_input
)
sd.wait()

write("temp_input.wav", sample_rate, recording)

# ----------------------------
# Transcribe with Whisper
# ----------------------------
model = whisper.load_model("base")  # tiny, small, base, medium, large
result = model.transcribe("temp_input.wav")
user_text = result["text"].strip()
print("You said:", user_text)

# ----------------------------
# Generate lamp response
# ----------------------------
prompt = f"""
You are a gentle, reflective homework helper lamp. The student said: "{user_text}".
Give a thoughtful, encouraging response. Suggest rethinking strategies or hints, 
but do not give the full answer unless asked explicitly.
"""

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "system", "content": prompt}]
)

lamp_response = response.choices[0].message.content.strip()
print("Lamp says:", lamp_response)

# ----------------------------
# Speak the response
# ----------------------------
speak(lamp_response)
