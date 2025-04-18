import streamlit as st
import tempfile
import os
from fina_pipeline import compiled_graph

st.set_page_config(page_title="🧠 Multilingual Meeting Summarizer", layout="centered")
st.title("🧠 AI-Powered Meeting Summarizer")

st.markdown("""
Upload your meeting audio file (MP3/WAV). The app will:
1. Denoise the audio 🔊  
2. Detect the language (English, Mandarin, Cantonese) 🌐  
3. Generate a clean summary 📝
""")

# File upload
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(tmp_path)
    st.info("🔄 Processing audio... Please wait...")

    try:
        result = compiled_graph.invoke({"audio_path": tmp_path})
        summary = result["summary"]

        st.success("✅ Summary generated!")
        st.markdown("### 📝 Meeting Summary")
        st.markdown(summary)

        st.download_button(
            label="📄 Download Summary",
            data=summary,
            file_name="meeting_minutes.md",
            mime="text/markdown"
        )
    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
    finally:
        os.remove(tmp_path)
