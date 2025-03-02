

---

# Auto ASR & Speaker Diarization for Telesales Calls (Demo)

A prototype application that automatically analyzes and summarizes audio conversations (e.g., telesales calls) by leveraging Speaker Diarization and Automatic Speech Recognition (ASR).

---

## 🛠️ Key Features

- **Audio File Uploading** (supports WAV files)
- **Automatic Speaker Diarization** (speaker identification and segmentation) using `pyannote.audio`
- **Audio Segmentation** based on identified speakers with merging of closely spaced segments
- **Automatic Speech Recognition (ASR)** for Thai language transcription using NECTEC's Whisper Thai model
- **Interactive Chat Log with Audio Playback**
- **Automatic Conversation Summarization** using a Large Language Model (LLM) via Ollama
- **Silence Metrics Calculation** (Total Silence duration and Longest Silence)
- **Speaker Distribution Analysis** presented as an interactive Pie Chart

---

## ⚙️ Technologies and Libraries Used

- [Streamlit](https://streamlit.io/) (Frontend UI)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) (Speaker Diarization)
- [Whisper (NECTEC Pathumma)](https://huggingface.co/nectec/Pathumma-whisper-th-large-v3) (Thai ASR)
- [pydub](https://github.com/jiaaro/pydub) (Audio Processing)
- [LangChain Ollama](https://python.langchain.com/docs/integrations/llms/ollama) (Conversation Summarization)
- [Llama3.2:3b or other LLMs via Ollama](https://ollama.ai/library/llama3) (Language Models)
- PyTorch (CUDA GPU acceleration support)

---

## 🚀 Installation and Usage

### 1. Clone the Repository and Install Dependencies

```bash
git clone https://github.com/iristap/Auto-ASR-Diarization-Telesales-call.git
cd Auto-ASR-Diarization-Telesales-call
```

**Note:**  
- Ollama and desired LLM models must be pre-installed.
- HuggingFace Token is required for accessing the diarization model.

### 2. Setup Environment Variables

Create a `.env` file in your project root folder:

```bash
HF_Token_Read=YOUR_HUGGINGFACE_TOKEN
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

Open your browser at the provided URL (`localhost:8501` by default).

---

## 🖥️ Application Workflow

1. **Upload** your WAV audio file via Streamlit UI.
2. **Diarization** will automatically identify and segment speakers.
3. **ASR transcription** will be performed on each audio segment.
4. **Interactive Chat Log** is displayed with audio playback for each segment.
5. **Automatic Conversation Summary** is generated using an LLM.
6. **Additional analytics** like Silence Metrics and Speaker Distribution Pie Chart are displayed.

---

## 📌 Example Screenshots

![image](https://github.com/user-attachments/assets/8af988c9-655a-4595-8ab6-c7b3888c4928)

---
