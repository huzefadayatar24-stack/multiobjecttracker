import streamlit as st
import cv2
import tempfile
import os
import subprocess
from ultralytics import YOLO
import supervision as sv

# ---------- PAGE SETUP & CUSTOM UI ----------
st.set_page_config(page_title="Vision MOT Dashboard", layout="wide", page_icon="👁️")

st.markdown("""
<style>
    .stApp { font-family: 'Space Grotesk', sans-serif; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #f2a65a; margin-bottom: 0px; }
    .sub-header { color: #8b96a3; font-size: 1.1rem; margin-bottom: 2rem; }
    .stButton>button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: 600; 
        background-color: #f2a65a; 
        color: #12171d; 
        border: none;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover { background-color: #f5b675; }
    .stProgress > div > div > div > div { background-color: #f2a65a; }
    div[data-testid="stMetricValue"] { color: #f2a65a; font-family: 'IBM Plex Mono', monospace; }
    
    [data-testid="stVideo"] {
        display: flex;
        justify-content: center;
        background-color: #0b0e12;
        border-radius: 8px;
        padding: 10px;
    }
    [data-testid="stVideo"] video {
        max-height: 60vh; 
        width: auto !important; 
        max-width: 100%;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Vision MOT Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ultra-Fast Backend CPU Tracking Engine</p>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  

model = load_model()
all_classes = {0: "person", 2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}

# ---------- DASHBOARD LAYOUT ----------
col_controls, col_stage = st.columns([1, 2.5], gap="large")

with col_controls:
    st.subheader("⚙️ Tracking Parameters")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    iou_threshold = st.slider("IoU Threshold", 0.1, 1.0, 0.5, 0.05)
    
    selected_labels = st.multiselect(
        "Active Classes",
        options=list(all_classes.values()),
        default=list(all_classes.values())
    )
    selected_class_ids = [k for k, v in all_classes.items() if v in selected_labels]

    st.divider()
    st.subheader("📁 Input Source")
    uploaded_file = st.file_uploader("Drop a video file here", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    start_btn = st.button("▶️ Process Video Fast")

with col_stage:
    stage_header = st.empty()
    status_text = st.empty()
    progress_bar = st.empty()
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if not uploaded_file:
        stage_header.subheader("Stage: Standby")
        status_text.info("Upload a video and click Process.")

# ---------- PROCESSING LOGIC ----------
if uploaded_file is not None and start_btn:
    
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_file.read())
    input_path = input_temp.name

    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1. OPTIMIZATION: Display Downscale
    max_width = 480
    if width > max_width:
        scale = max_width / width
        width, height = int(width * scale), int(height * scale)

    # 2. OPTIMIZATION: Temporal Downscale (Process at half FPS)
    target_fps = original_fps / 2.0

    output_path = os.path.join(tempfile.gettempdir(), "raw_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    
    unique_counts = {}  

    stage_header.subheader("⚡ Processing Video at High Speed...")
    status_text.info("Running optimizations: Frame skipping + 320px inference tensor.")
    prog_bar = progress_bar.progress(0)

    frame_idx = 0
    frames_processed = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        
        # Skip every other frame (cuts workload by 50%)
        if frame_idx % 2 != 0:
            continue
            
        frames_processed += 1
        frame = cv2.resize(frame, (width, height))

        # 3. OPTIMIZATION: Inference Downscale (imgsz=320 cuts CPU math by 75%)
        results = model(frame, conf=conf_threshold, iou=iou_threshold, classes=selected_class_ids, imgsz=320, verbose=False)[0]
        
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = []
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            class_name = all_classes.get(class_id, "obj")
            labels.append(f"#{tracker_id} {class_name}")
            if class_name not in unique_counts:
                unique_counts[class_name] = set()
            unique_counts[class_name].add(tracker_id)

        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        writer.write(annotated)

        # Update progress bar smoothly based on total frames
        if total_frames > 0 and frame_idx % 10 == 0:
            prog_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    
    # ----- 4. OPTIMIZATION: FFMPEG Ultrafast Preset -----
    stage_header.subheader("Finalizing Video...")
    status_text.info("Running ultrafast H.264 compression...")
    prog_bar.progress(1.0)
    
    h264_output_path = os.path.join(tempfile.gettempdir(), "h264_output.mp4")
    
    try:
        # Added -preset ultrafast to encode the video almost instantly
        subprocess.run(["ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264", "-preset", "ultrafast", "-crf", "28", h264_output_path], 
                       check=True, capture_output=True, text=True)
        final_video_path = h264_output_path
    except Exception as e:
        final_video_path = output_path
        st.error("FFmpeg conversion failed. Ensure 'ffmpeg' is in packages.txt.")

    unique_counts = {label: len(ids) for label, ids in unique_counts.items()}
    
    stage_header.subheader("✅ Tracking Complete")
    status_text.empty()
    prog_bar.empty() 
    
    with video_placeholder.container():
        st.video(final_video_path)
    
    with stats_placeholder.container():
        st.divider()
        st.subheader("📊 Lifetime Object Counts (Unique IDs)")
        if unique_counts:
            metric_cols = st.columns(len(unique_counts))
            for i, (label, count) in enumerate(unique_counts.items()):
                metric_cols[i].metric(label.capitalize() + "s", count)
        else:
            st.info("No tracked objects detected in this sequence.")

    with col_controls:
        with open(final_video_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Annotated Video", 
                data=f, 
                file_name="tracked_output.mp4",
                mime="video/mp4",
                type="primary"
            )
