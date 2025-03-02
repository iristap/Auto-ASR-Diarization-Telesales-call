import streamlit as st
import os
import tempfile
import torch

# pyannote สำหรับ Speaker Diarization
from pyannote.audio import Pipeline

# transformers สำหรับ ASR
from transformers import pipeline as hf_pipeline

# pydub สำหรับจัดการไฟล์เสียง
from pydub import AudioSegment

#########################
# 1) ส่วน Config & Setup
#########################
st.title("Speaker Diarization + Thai Transcription (Chat Style)")

@st.cache_resource
def load_diarization_pipeline():
    """โหลดโมเดล pyannote สำหรับ Speaker Diarization"""
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_Token_Read")  # ถ้าโมเดลเป็น public อาจไม่ต้องใส่ token
    )
    if torch.cuda.is_available():
        diar_pipeline.to(torch.device("cuda"))
    return diar_pipeline

@st.cache_resource
def load_asr_pipeline():
    """
    โหลดโมเดล ASR สำหรับภาษาไทย (Whisper) จาก Hugging Face
    ตัวอย่างใช้ nectec/Pathumma-whisper-th-large-v3
    """
    lang = "th"
    task = "transcribe"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = hf_pipeline(
        task="automatic-speech-recognition",
        model="nectec/Pathumma-whisper-th-large-v3",
        torch_dtype="auto",
        device=device
    )
    # บังคับให้โมเดลถอดเป็นภาษาไทย
    pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language=lang, task=task)
    return pipe

diar_pipeline = load_diarization_pipeline()
asr_pipeline = load_asr_pipeline()

#########################
# 2) ส่วนอัพโหลดไฟล์เสียง
#########################
uploaded_file = st.file_uploader("อัพโหลดไฟล์เสียง (WAV เท่านั้น)", type=["wav"])
if uploaded_file is not None:
    # เขียนไฟล์ชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success(f"อัพโหลดไฟล์เรียบร้อย: {tmp_path}")

    #########################
    # 3) Diarization
    #########################
    st.write("กำลังประมวลผล Speaker Diarization ...")
    diarization = diar_pipeline(tmp_path, num_speakers=2)  # กำหนด num_speakers=2 (ปรับได้)

    # เก็บ segment (start, end, speaker)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))

    # แสดงผล segment
    st.write("Segments ที่ได้จาก Diarization:")
    for seg in segments:
        # st.write(f"start={seg[0]:.2f}s  end={seg[1]:.2f}s  speaker_{seg[2]}")
        st.write(f"speaker_{seg[2]}: {seg[0]:.2f}s - {seg[1]:.2f}s")

    #########################
    # 4) Merge segments + ตัดไฟล์เสียง + ASR
    #########################
    audio_file = AudioSegment.from_wav(tmp_path)

    # ตั้งค่า padding เพื่อลดการตัดคำ
    PADDING = 0.2  # วินาที
    # ตั้งค่า merge_threshold เพื่อรวม segment ที่อยู่ใกล้กัน
    MERGE_THRESHOLD = 0.5  # วินาที

    def merge_close_segments(seg_list, merge_threshold=0.5):
        merged = []
        for seg in seg_list:
            start, end, spk = seg
            if not merged:
                merged.append(seg)
            else:
                last_start, last_end, last_spk = merged[-1]
                if (spk == last_spk) and (start - last_end <= merge_threshold):
                    merged[-1] = (last_start, end, last_spk)
                else:
                    merged.append(seg)
        return merged

    segments_merged = merge_close_segments(segments, MERGE_THRESHOLD)

    st.write("กำลังถอดเสียงภาษาไทย (ASR) ...")
    conversation = []

    for seg in segments_merged:
        start, end, spk = seg

        # เพิ่ม padding
        start_pad = max(0, start - PADDING)
        end_pad = min(len(audio_file) / 1000, end + PADDING)

        start_ms = int(start_pad * 1000)
        end_ms = int(end_pad * 1000)

        chunk_audio = audio_file[start_ms:end_ms]

        # เขียนไฟล์ชั่วคราว
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as chunk_tmp:
            chunk_audio.export(chunk_tmp.name, format="wav")
            chunk_path = chunk_tmp.name

        # ถอดเสียงด้วย ASR
        result = asr_pipeline(chunk_path, return_timestamps=True)
        text = result["text"]

        # สมมติ speaker_00 = Agent, speaker_01 = Customer
        role = "Agent" if spk == "SPEAKER_00" else "Customer"
        conversation.append((role, text))

        # ลบไฟล์ชั่วคราว
        os.remove(chunk_path)

    st.success("ถอดเสียงเสร็จสิ้น!")

    #########################
    # 5) แสดงผลในรูปแบบ "แชท"
    #########################
    st.header("Conversation")
    for (role, text) in conversation:
        if role == "Agent":
            # แสดง Agent เป็น assistant
            with st.chat_message("assistant"):
                st.write(f"**Agent**: {text}")
        else:
            # แสดง Customer เป็น user
            with st.chat_message("user"):
                st.write(f"**Customer**: {text}")
