# 🧠 Multilingual AI-Powered Meeting Summarizer

This Streamlit application enables users to **upload meeting audio** in English, Mandarin, or Cantonese and receive a **clean, structured summary**. The backend pipeline performs denoising, language detection, transcription, and summary generation using Whisper, LangGraph, and Ollama's `deepseek` model.

---

## 🚀 Features

- 🔊 **Audio Denoising**: Reduces background noise and other disturbances  using `noisereduce` for effective speech recognition
- 🌐 **Language Detection**: Supports **English**, **Mandarin**, and **Cantonese**
- 🗣 **Speech-to-Text**: Uses Whisper via HuggingFace Transformers
- 📝 **Meeting Summarization**: Structured into:
  - Key Discussion Points
  - Takeaways
  - Action Items
- 📄 **Markdown Export**: Download the final summary as a `.md` file

---



## 📂 Main Components

| File                  | Purpose |
|-----------------------|---------|
| `app.py`              | Main UI built with Streamlit for audio upload and displaying results |
| `./samples`           | contains test results|
| `project_structure.md`| Structural overview of the project layout |
| `requirements.txt`    | List of Python dependencies required for the project |
| `README.md`           | Project overview, setup instructions, and usage details |


---
## 📦 Installation Steps


## 🧾 Prerequisites

Make sure the following are installed on your machine:

- Python 3.8 or higher
- `git`
- `ffmpeg` (required for audio processing)
- Virtual environment tool (optional but recommended)

  ### 1. Clone the Repository

```bash
git clone https://github.com/BayesianDecoder/multiagent_for_call_conversation.git
cd multiagent_for_call_conversation
```
### 2. Create & Activate a Virtual Environment (Optional but Recommended)

``` bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\\Scripts\\activate
```
### 3.  Install Python Dependencies

``` bash
pip install -r requirements.txt
```

### 🚀 Running the App

``` bash
streamlit run app.py
```
- This will launch the UI at: `http://localhost:8501`
- You can now upload .mp3 or .wav files and receive summarized meeting notes.

### 🧪 Testing the Pipeline

To test the backend pipeline directly:

``` bash
python fina_pipeline.py
```
- Make sure the default audio path inside the script is valid or modify it accordingly.

  ### 🧰 Additional Tips
Ensure GPU or MPS support is available if running Whisper for faster transcription
