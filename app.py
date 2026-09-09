import streamlit as st
import cv2
import tempfile
import os
import subprocess
from ultralytics import YOLO
import supervision as sv

# ---------- PAGE SETUP & STYLING ----------
st.set_page_config(page_title="High-Fidelity MOT Dashboard", layout="wide", page_icon="🎯")

st.markdown("""
<style>
    .stApp { font-family: 'Space Grotesk', sans-serif; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #f2a65a; margin-bottom: 0px; }
    .sub-header { color: #8b96a3; font-size: 1.05rem; margin-bottom: 1.5rem; }
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
        padding: 8px;
    }
    [data-testid="stVideo"] video {
        max-height: 65vh; 
        width: auto !important; 
        max-width: 100%;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">High-Fidelity Object Tracker</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Full Precision Detection & Persistent Association Engine</p>', unsafe_allow_html=True)

# ---------- SIDEBAR & CONTROLS ----------
col_controls, col_stage = st.columns([1, 2.5], gap="large")

with col_controls:
    st.subheader("⚙️ Model Configuration")
    
    # Model Selector: Nano for testing, Small for full accuracy
    model_choice = st.selectbox(
        "Detection Model",
        options=["yolov8s.pt (Recommended - High Accuracy)", "yolov8n.pt (Faster - Lower Precision)"],
        index=0
    )
    selected_model_path = "yolov8s.pt" if "yolov8s" in model_choice else "yolov8n.pt"

    @st.cache_resource
    def get_model(path):
        return YOLO(path)
    
    model = get_model(selected_model_path)
    
    st.divider()
    st.subheader("🎯 Tracking Parameters")
    
    # Optimal detection threshold for ByteTrack: 0.20 - 0.25
    conf_threshold = st.slider("Detection Confidence", 0.10, 0.90, 0.25, 0.05,
                               help="Lower values allow ByteTrack to detect distant and partially occluded objects.")
    iou_threshold = st.slider("NMS IoU Threshold", 0.20, 0.80, 0.50, 0.05)
    
    all_classes = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    selected_labels = st.multiselect(
        "Classes to Track",
        options=list(all_classes.values()),
        default=list(all_classes.values())
    )
    selected_class_ids = [k for k, v in all_classes.items() if v in selected_labels]

    st.divider()
    st.subheader("📁 Input Video")
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    start_btn = st.button("▶️ Run High-Quality Tracking")

with col_stage:
    stage_header = st.empty()
    status_text = st.empty()
    progress_bar = st.empty()
    video_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    if not uploaded_file:
        stage_header.subheader("Stage: Standby")
        status_text.info("Upload a video and select your desired model to begin.")

# ---------- PROCESSING PIPELINE ----------
if uploaded_file is not None and start_btn:
    
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_file.read())
    input_path = input_temp.name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Keep native resolution to preserve every pixel for distant objects
    output_path = os.path.join(tempfile.gettempdir(), "raw_hd_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Tuned ByteTrack configuration
    tracker = sv.ByteTrack(
        track_activation_threshold=conf_threshold,
        lost_track_buffer=45,         # Keep track ID alive for 45 frames during occlusions
        minimum_matching_threshold=0.8
    )
    
    # Clean, professional annotation styling
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_scale=0.45, 
        text_thickness=1, 
        text_padding=4
    )
    
    unique_counts = {}

    stage_header.subheader(f"⚙️ Tracking Active ({selected_model_path})")
    status_text.info("Analyzing full-resolution frames. Maintaining spatial & temporal consistency...")
    prog_bar = progress_bar.progress(0)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference at 640px native letterbox without squashing aspect ratio
        results = model(
            frame, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            classes=selected_class_ids, 
            imgsz=640, 
            verbose=False
        )[0]
        
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = []
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            class_name = all_classes.get(class_id, "obj")
            labels.append(f"#{tracker_id} {class_name}")

            if class_name not in unique_counts:
                unique_counts[class_name] = set()
            unique_counts[class_name].add(tracker_id)

        # Render annotations directly onto original resolution frame
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        writer.write(annotated)

        frame_idx += 1
        if total_frames > 0 and frame_idx % 4 == 0:
            prog_bar.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    
    # ---------- LOSSLESS WEB CONVERSION ----------
    stage_header.subheader("Encoding High-Quality Video...")
    status_text.info("Applying H.264 high-profile encoding...")
    prog_bar.progress(1.0)
    
    h264_output_path = os.path.join(tempfile.gettempdir(), "h264_hd_output.mp4")
    
    try:
        # CRF 18 = Visually lossless, preset medium = crisp edges and text
        subprocess.run([
            "ffmpeg", "-y", "-i", output_path,
            "-vcodec", "libx264",
            "-crf", "18",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            h264_output_path
        ], check=True, capture_output=True, text=True)
        final_video_path = h264_output_path
    except Exception as e:
        final_video_path = output_path
        st.error(f"FFmpeg conversion fallback engaged: {e}")

    unique_counts = {label: len(ids) for label, ids in unique_counts.items()}
    
    # UI Reset & Display
    stage_header.subheader("✅ High-Fidelity Tracking Complete")
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
            st.info("No tracked objects detected.")

    with col_controls:
        with open(final_video_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Full-Quality Video", 
                data=f, 
                file_name="tracked_output_hd.mp4",
                mime="video/mp4",
                type="primary"
            )
