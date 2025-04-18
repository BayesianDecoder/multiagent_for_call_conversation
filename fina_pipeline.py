import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
import noisereduce as nr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
import ollama
import re
import os
import langdetect

# === Node 1: Audio Denoising / Preprocessing ===
def denoise_audio_node(state):
    audio_path = state["audio_path"]
    print("[Node 0] Loading and denoising audio...")

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

    # Apply noise reduction
    reduced = nr.reduce_noise(
        y=samples,
        sr=16000,
        stationary=True,
        prop_decrease=0.75
    )

    # Save cleaned audio to WAV for reuse using soundfile 
    import soundfile as sf
    cleaned_path = "cleaned_audio.wav"
    sf.write(cleaned_path, reduced, 16000)

    print("[Node 0] Audio denoised and saved.")
    return {"cleaned_audio_path": cleaned_path}

# === Load Whisper Model ===
AUDIO_MODEL = "openai/whisper-medium"
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to: {device}")
model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL, torch_dtype=torch.float32).to(device)
processor = AutoProcessor.from_pretrained(AUDIO_MODEL)

# === Helper Function ===
def load_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform.to(device).squeeze(0), sample_rate

# === Node 2: Language Detection & Transcription ===
def detect_language_and_transcribe_node(state):
    audio_path = state["cleaned_audio_path"]
    speech, sample_rate = load_audio(audio_path)

    inputs = processor(speech.cpu(), sampling_rate=sample_rate, return_tensors="pt").to(device)

    print("[Node 1] Performing language detection using Whisper...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False
        )

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if any("你好" in word or "我们" in word for word in transcription):
        detected_lang = "mandarin"
    elif any("咩" in word or "冇" in word for word in transcription):
        detected_lang = "cantonese"
    else:
        detected_lang = "english"

    print(f"[Node 1] Detected Language: {detected_lang}")
    return {"transcription": transcription, "detected_language": detected_lang}

# === Node 3: Generate Summary in Detected Language ===
def generate_summary_node(state):
    transcription = state["transcription"]
    lang = state["detected_language"]

    print(f"[Node 2] Generating summary in language: {lang}")
    system_message = """You are an expert assistant specializing in generating concise and actionable meeting minutes from audio transcripts. Your goal is to extract meaningful insights, key discussions, and actionable next steps from the provided text."""

    user_prompt = f"""Generate meeting minutes from the following transcript. Focus on extracting:

1. Key Discussion Points
- Main topics discussed
- Important insights
- Significant conversations

2. Takeaways
- Core learnings
- Critical insights
- Strategic implications

3. Action Items
- Specific tasks or next steps
- Prioritize clear, actionable items
- Include any suggested responsibilities (if mentioned)

Transcript:
{transcription}

Guidelines:
- Be concise and precise
- Use markdown formatting
- Include dates, locations, or names only if they are explicitly mentioned 
- Prioritize actionable information
- If information is unclear or missing, note it appropriately
- Preserve original language: {lang}"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    MODEL = 'deepseek-r1'
    response = ollama.chat(model=MODEL, messages=messages)
    result = response['message']['content']
    summary = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
    return {"summary": summary}

# === Graph State ===
class GraphState(TypedDict):
    audio_path: str
    cleaned_audio_path: str
    transcription: str
    detected_language: str
    summary: str

# === Build LangGraph ===
graph = StateGraph(GraphState)
graph.add_node("DenoiseAudio", denoise_audio_node)
graph.add_node("DetectLangAndTranscribe", detect_language_and_transcribe_node)
graph.add_node("GenerateSummary", generate_summary_node)

graph.set_entry_point("DenoiseAudio")
graph.add_edge("DenoiseAudio", "DetectLangAndTranscribe")
graph.add_edge("DetectLangAndTranscribe", "GenerateSummary")
graph.add_edge("GenerateSummary", END)

compiled_graph = graph.compile()

# === Run ===
if __name__ == "__main__":
    audio_file = "/Users/vijay/Documents/PROJECTS ML/Spotify recommender systerm/AI-Meeting-Minutes-Generator-main/3.mp3"
    result = compiled_graph.invoke({"audio_path": audio_file})
    print("\n\nFinal Summary:\n")
    print(result["summary"])
    with open("final_meeting_minutes.md", "w", encoding="utf-8") as f:
        f.write(result["summary"])