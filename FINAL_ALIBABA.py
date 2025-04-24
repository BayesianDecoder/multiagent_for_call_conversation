

import numpy as np
from pydub import AudioSegment
import noisereduce as nr
import soundfile as sf
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict
import ollama
import re
import langdetect
import time

# === Alibaba ASR Credentials ===
# === Load Environment Variables ===
load_dotenv()

APP_KEY = os.getenv("ALIBABA_APP_KEY")
ACCESS_TOKEN = os.getenv("ALIBABA_ACCESS_TOKEN")

API_ENDPOINT = "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/file"
API_ENDPOINT = "https://nls-gateway-cn-shanghai.aliyuncs.com/stream/v1/file"

# === Configure Retry Strategy ===
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)

# === Node 1: Audio Denoising/Preprocessing ===
def denoise_audio_node(state):
    audio_path = state["audio_path"]
    print("[Node 0] Loading and processing audio...")

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    cleaned_path = "cleaned_audio.flac"
    sf.write(cleaned_path, samples, 16000, format='FLAC')

    print(f"[Node 0] Audio processed and saved as {cleaned_path}")
    return {"cleaned_audio_path": cleaned_path}

# === Node 2: Transcription with Enhanced Error Handling ===
def detect_language_and_transcribe_node(state):
    audio_path = state["cleaned_audio_path"]
    file_format = "flac"  # Using FLAC for better compression

    params = {
        "appkey": APP_KEY,
        "token": ACCESS_TOKEN,
        "format": file_format,
        "sample_rate": 16000  # As integer
    }

    headers = {
        "Content-Type": "application/octet-stream",
        "X-NLS-Enable-VAD": "true"  # Enable voice activity detection
    }

    print(f"[Node 1] Sending {os.path.getsize(audio_path)/1024:.1f}KB audio to ASR...")
    
    try:
        with open(audio_path, 'rb') as f:
            response = http.post(
                API_ENDPOINT,
                params=params,
                headers=headers,
                data=f,
                timeout=(10, 30)  # Connect/read timeouts
            )

        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "20000000":
                transcription = result.get("result", "")
            else:
                print(f"ASR Error: {result.get('status')} - {result.get('message')}")
                transcription = ""
        else:
            print(f"HTTP Error {response.status_code}: {response.text}")
            transcription = ""

    except Exception as e:
        print(f"ASR Request Failed: {str(e)}")
        transcription = ""

    # Language detection with fallback
    if transcription.strip():
        try:
            lang_code = langdetect.detect(transcription)
            lang_map = {
                "zh-cn": "mandarin",
                "yue": "cantonese",
                "en": "english"
            }
            detected_lang = lang_map.get(lang_code, "unknown")
        except:
            detected_lang = "unknown"
    else:
        detected_lang = "unknown"

    print(f"[Node 1] Transcription: {transcription[:80]}...")
    print(f"[Node 1] Detected Language: {detected_lang}")
    return {"transcription": transcription, "detected_language": detected_lang}

# === Node 3: Summary Generation === 
def generate_summary_node(state):
    transcription = state["transcription"]
    lang = state["detected_language"]
    
    print(f"[Node 2] Generating summary in {lang}...")
    
    prompt = f"""Generate structured meeting minutes from this transcript:
{transcription}

Include:
1. Key Decisions - bullet points
2. Action Items - with owners and deadlines
3. Discussion Highlights - main topics
4. Next Steps - clear deliverables

Format in markdown. Preserve {lang} language."""
    
    try:
        response = ollama.chat(
            model='deepseek-r1:1.5b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3}
        )
        summary = response['message']['content']
        summary = re.sub(r'<.*?>', '', summary)  # Remove any HTML tags
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        summary = "Summary generation failed"
    
    return {"summary": summary}

# === State Graph Setup ===
class GraphState(TypedDict):
    audio_path: str
    cleaned_audio_path: str
    transcription: str
    detected_language: str
    summary: str

graph = StateGraph(GraphState)
graph.add_node("DenoiseAudio", denoise_audio_node)
graph.add_node("DetectLangAndTranscribe", detect_language_and_transcribe_node)
graph.add_node("GenerateSummary", generate_summary_node)

graph.set_entry_point("DenoiseAudio")
graph.add_edge("DenoiseAudio", "DetectLangAndTranscribe")
graph.add_edge("DetectLangAndTranscribe", "GenerateSummary")
graph.add_edge("GenerateSummary", END)

compiled_graph = graph.compile()

# === Execution ===
if __name__ == "__main__":
    # Replace with your actual audio path
    audio_file = "/Users/vijay/Documents/PROJECTS/Spotify recommender systerm/AI-Meeting-Minutes-Generator-main/final /samples/cantonese/chinese1.mp3"  
    
    start_time = time.time()
    result = compiled_graph.invoke({"audio_path": audio_file})
    duration = time.time() - start_time
    
    print(f"\nProcessing completed in {duration:.1f}s")
    print("\nMeeting Minutes:")
    print(result["summary"])
    
    with open("minutes.md", "w") as f:
        f.write(result["summary"])
    print("\nSaved to minutes.md")