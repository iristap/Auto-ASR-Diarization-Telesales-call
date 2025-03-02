import streamlit as st
import os
import tempfile
import torch
from pyannote.audio import Pipeline
from transformers import pipeline as hf_pipeline
from pydub import AudioSegment
import shutil
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

###############################
# Helper functions
###############################

def format_time(seconds):
    """แปลงเวลา (วินาที) เป็นรูปแบบ H:mm:ss"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"

def preprocess_and_segment_audio(audio_path, segments, output_folder="segmented_audio", merge_threshold=1, padding=0.5, save_in_one_folder=True):
    """
    Preprocesses speaker diarization segments by merging close segments and splits audio accordingly with optional padding.
    
    Parameters:
      - audio_path (str): Path to the audio file.
      - segments (list of tuples): List of (start, end, speaker) segments.
      - output_folder (str): Folder to save segmented audio files.
      - merge_threshold (float): Maximum gap (in seconds) between segments to merge.
      - padding (float): Additional time (in seconds) before and after each segment.
      - save_in_one_folder (bool): หาก True ทุกไฟล์จะเก็บในโฟลเดอร์เดียว
     
    Returns:
      - saved_files (list): List of tuples (segment_file_path, start, end, speaker)
      - merged_segments (list): List ของ merged segments
    """
    # โหลดไฟล์เสียงด้วย pydub
    audio = AudioSegment.from_wav(audio_path)
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    # สร้าง output folder หากยังไม่มี
    os.makedirs(output_folder, exist_ok=True)
    
    # Merge segments ที่อยู่ใกล้กันและเป็น speaker เดียวกัน
    merged_segments = []
    # for segment in segments:
    #     start, end, speaker = segment
    #     if merged_segments and speaker == merged_segments[-1][2]:
    #         prev_start, prev_end, prev_speaker = merged_segments[-1]
    #         if start - prev_end <= merge_threshold:
    #             merged_segments[-1] = (prev_start, end, prev_speaker)
    #         else:
    #             merged_segments.append(segment)
    #     else:
    #         merged_segments.append(segment)
    # Merge segments โดยเพิ่มเงื่อนไขกรณีเกิด intersection
    merged_segments = []
    for segment in segments:
        start, end, speaker = segment
        if merged_segments:
            prev_start, prev_end, prev_speaker = merged_segments[-1]
            # st.write(f"prev_start={prev_start}s, prev_end={prev_end}s, prev_speaker={prev_speaker}")
            # st.write(f"start={start}s, end={end}s, speaker={speaker}")
            if start <= prev_end:
                # print(f"Skipping overlapping segments: {prev_speaker} vs {speaker}")
                # st.write(f"Skipping overlapping segments: {prev_speaker} vs {speaker}")
                # มีการทับซ้อนกัน
                if speaker != prev_speaker:
                    # ถ้า speaker ต่างกัน ให้ข้าม segment ปัจจุบัน (ไม่เก็บ)
                    # print(f"Skipping overlapping segments: {prev_speaker} vs {speaker}")
                    # st.write(f"Skipping overlapping segments: {prev_speaker} vs {speaker}")
                    continue
                else:
                    # ถ้า speaker เหมือนกัน แม้ทับซ้อนกัน ก็ไม่ merge แต่เก็บแยก segment ใหม่
                    merged_segments.append(segment)
            else:
                # ไม่มีการทับซ้อนกัน
                if speaker == prev_speaker and (start - prev_end) <= merge_threshold:
                    # ถ้า gap ไม่เกิน merge_threshold ให้ merge (กรณีไม่ทับซ้อน)
                    merged_segments[-1] = (prev_start, end, speaker)
                else:
                    merged_segments.append(segment)
        else:
            merged_segments.append(segment)

    
    saved_files = []
    for i, (start, end, speaker) in enumerate(merged_segments):
        # เพิ่ม padding ที่เริ่มต้นและสิ้นสุด
        start = max(0, start - padding)
        end = min(len(audio)/1000, end + padding)
        
        # แปลงเวลาเป็นมิลลิวินาที
        start_ms = int(start * 1000)
        end_ms = int(end * 1000)
        
        # ตัดไฟล์เสียง
        segment_audio = audio[start_ms:end_ms]
        
        # หากเก็บในโฟลเดอร์เดียวกัน
        if save_in_one_folder:
            speaker_folder = output_folder
        else:
            speaker_folder = os.path.join(output_folder, speaker)
            os.makedirs(speaker_folder, exist_ok=True)
        
        # กำหนดชื่อไฟล์ และ export ไฟล์เสียง segment
        segment_path = os.path.join(speaker_folder, f"{speaker}_{i+1:03d}_{start:.2f}s-{end:.2f}s.wav")
        segment_audio.export(segment_path, format="wav")
        # st.write(f"Saved: {segment_path}")
        saved_files.append((segment_path, start, end, speaker))
    
    st.success("Finished segmenting audio.")
    return saved_files, merged_segments

###############################
# Streamlit App
###############################

st.title("Demo Auto ASR Diarization Telesales call")

# Step 1. อัพโหลดไฟล์เสียง
uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])
if uploaded_file is not None:
    # บันทึกไฟล์เสียงลงไฟล์ชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name
    st.success("Uploaded audio file.")

    ###############################
    # Step 2. Diarization ด้วย pyannote
    ###############################
    st.info("Loading diarization model...")
    diar_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_Token_Read")
    )
    if torch.cuda.is_available():
        diar_pipeline.to(torch.device("cuda"))
    st.success("Diarization model loaded.")

    st.info("Running diarization...")
    # กำหนดจำนวน speaker (ปรับตามความเหมาะสม)
    diarization = diar_pipeline(audio_path, min_speakers=2, max_speakers=5)
    list_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        list_segments.append((turn.start, turn.end, speaker))
    st.success("Diarization finished.")

    ###############################
    # Step 3. Preprocess & Segment Audio
    ###############################
    saved_files, merged_segments = preprocess_and_segment_audio(audio_path, list_segments, output_folder="segmented_audio", merge_threshold=1.5, padding=0.5)

    ###############################
    # Step 4. สร้าง Chat Log พร้อม Audio Player และ ASR
    ###############################
    st.header("Chat Log with Audio and ASR")
    
    st.info("Loading ASR model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lang = "th"
    task = "transcribe"
    asr_pipeline = hf_pipeline(
        task="automatic-speech-recognition",
        model="nectec/Pathumma-whisper-th-large-v3",
        torch_dtype="auto",
        device=device
    )
    asr_pipeline.model.config.forced_decoder_ids = asr_pipeline.tokenizer.get_decoder_prompt_ids(language=lang, task=task)
    st.success("ASR model loaded.")

    
    # ตัวแปรสำหรับเก็บข้อความที่รวมกันของแต่ละกลุ่ม speaker
    aggregated_chat = []

    for segment_path, start, end, speaker in saved_files:
        # st.write(segment_path)
        st.markdown(f"**{speaker} [{format_time(start)} - {format_time(end)}]:**")
        
        # แสดง audio player
        with open(segment_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")
        
        # ทำ ASR กับไฟล์เสียง segment นี้
        asr_result = asr_pipeline(segment_path, return_timestamps=True)
        transcript = asr_result.get("text", "")
        st.write(transcript)
        
        # ถ้ากลุ่มสุดท้ายใน aggregated_chat เป็น speaker เดียวกัน ให้นำ transcript มาต่อ
        if aggregated_chat and aggregated_chat[-1]["speaker"] == speaker:
            aggregated_chat[-1]["transcript"] += " " + transcript
        else:
            aggregated_chat.append({"speaker": speaker, "transcript": transcript})

    # แสดงผลรวมในรูปแบบที่ต้องการ
    st.markdown("### Aggregated Chat")
    for entry in aggregated_chat:
        st.markdown(f"**{entry['speaker']}:** {entry['transcript']}")

    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    # llm = ChatOllama(model="deepseek-r1:8b", temperature=0)
  
    # รวมข้อความทั้งหมดจาก aggregated_chat เป็นข้อความเดียว
    full_chat = "\n".join([f"{entry['speaker']}: {entry['transcript']}" for entry in aggregated_chat])
    
    # สร้าง prompt สำหรับการสรุปบทสนทนา
    prompt_template = PromptTemplate(
        template="เขียนสรุปรายละเอียดของ Product จาก Chat log:\n{chat}\n:",
        input_variables=["chat"]
    )
    prompt = prompt_template.format(chat=full_chat)
    
    # เรียกใช้งาน ChatOllama เพื่อสรุปบทสนทนา
    summary = llm.invoke(prompt)
    
    st.markdown("### Call Summary")
    st.write(summary.content)

    # หลังจากแสดง Call Summary แล้ว
    st.markdown("### Silence Metrics")

    # โหลดไฟล์เสียงเต็มเพื่อหาความยาวทั้งหมด (ในหน่วยวินาที)
    audio_full = AudioSegment.from_wav(audio_path)
    call_duration = len(audio_full) / 1000.0  # เปลี่ยนเป็นวินาที

    # เรียงลำดับ segments ตามเวลาเริ่มต้น
    sorted_segments = sorted(merged_segments, key=lambda seg: seg[0])

    if sorted_segments:
        # คำนวณ silence ก่อน segment แรก
        total_silence = sorted_segments[0][0]
        # คำนวณ silence หลัง segment สุดท้าย
        total_silence += call_duration - sorted_segments[-1][1]
        
        # คำนวณ gap ระหว่าง segments
        gaps = []
        for i in range(1, len(sorted_segments)):
            gap = sorted_segments[i][0] - sorted_segments[i-1][1]
            if gap > 0:
                gaps.append(gap)
                total_silence += gap
        longest_silence = max(gaps) if gaps else max(sorted_segments[0][0], call_duration - sorted_segments[-1][1])
    else:
        total_silence = call_duration
        longest_silence = call_duration

    st.write(f"**Total Silence:** {format_time(total_silence)}")
    st.write(f"**Longest Silence:** {format_time(longest_silence)}")

    # คำนวณเวลาที่แต่ละ speaker พูดจาก merged_segments
    speaker_time = {}
    for (start, end, speaker) in merged_segments:
        duration = end - start
        speaker_time[speaker] = speaker_time.get(speaker, 0) + duration

    total_speaking_time = sum(speaker_time.values())
    speaker_percentage = {spk: (time/total_speaking_time)*100 for spk, time in speaker_time.items()}

    # แสดง pie chart สำหรับ % คำพูดของแต่ละ speaker
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    labels = list(speaker_percentage.keys())
    sizes = list(speaker_percentage.values())
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # ให้ pie chart เป็นวงกลม
    st.markdown("### Speaker Distribution")
    st.pyplot(fig)
