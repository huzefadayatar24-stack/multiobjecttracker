import streamlit as st
import cv2
import tempfile
import os
import subprocess
from ultralytics import YOLO
import supervision as sv

# ---------- PAGE SETUP & CUSTOM UI ----------
st.set_page_config(page_title="Vision MOT Dashboard", layout="wide", page_icon="👁️")

# Inject Custom CSS for a modern, web-app feel
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
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Vision MOT Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Live Multi-Object Tracking Engine</p>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  

model = load_model()
all_classes = {0: "person", 2: "car", 5: "bus", 7: "truck", 3: "motorcycle"}

# ---------- DASHBOARD LAYOUT ----------
# Split the screen into a control panel (left) and display stage (right)
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
    start_btn = st.button("▶️ Initialize Live Tracking")

with col_stage:
    # Placeholders that we will overwrite dynamically to create a live dashboard
    stage_header = st.empty()
    video_placeholder = st.empty()
    progress_bar = st.empty()
    stats_placeholder = st.empty()
    
    # Default idle state
    if not uploaded_file:
        stage_header.subheader("Live Feed: Standby")
        video_placeholder.info("Upload a video and click Initialize to begin live tracking.")

# ---------- PROCESSING LOGIC ----------
if uploaded_file is not None and start_btn:
    
    # Setup temporary files
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_file.read())
    input_path = input_temp.name

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Downscale for CPU performance
    max_width = 640
    if width > max_width:
        scale = max_width / width
        width, height = int(width * scale), int(height * scale)

    output_path = os.path.join(tempfile.gettempdir(), "raw_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize trackers and annotators
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    
    unique_counts = {}  

    # Update UI to active state
    stage_header.subheader("🔴 Live Feed: Active Tracking")
    prog_bar = progress_bar.progress(0, text="Initializing stream...")

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (width, height))

        # Run inference
        results = model(frame, conf=conf_threshold, iou=iou_threshold, classes=selected_class_ids, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = []
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
            class_name = all_classes.get(class_id, "obj")
            # Exact format from your image: "#ID class"
            labels.append(f"#{tracker_id} {class_name}")

            if class_name not in unique_counts:
                unique_counts[class_name] = set()
            unique_counts[class_name].add(tracker_id)

        # Draw annotations
        annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
        annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

        writer.write(annotated)

        # ----- LIVE RENDER EVERY FRAME -----
        # This replaces the static image with the newly annotated frame instantly
        video_placeholder.image(annotated, channels="BGR", use_container_width=True)

        frame_idx += 1
        if total_frames > 0:
            prog_bar.progress(min(frame_idx / total_frames, 1.0), text=f"Processing frame {frame_idx} / {total_frames}")

    cap.release()
    writer.release()
    
    # ----- POST-PROCESSING & FINAL EXPORT -----
    stage_header.subheader("Finalizing Video Export...")
    prog_bar.progress(1.0, text="Converting video format for web download...")
    
    h264_output_path = os.path.join(tempfile.gettempdir(), "h264_output.mp4")
    
    try:
        subprocess.run(["ffmpeg", "-y", "-i", output_path, "-vcodec", "libx264", h264_output_path], 
                       check=True, capture_output=True, text=True)
        final_video_path = h264_output_path
    except Exception as e:
        final_video_path = output_path
        st.error("FFmpeg conversion failed. Ensure 'ffmpeg' is in packages.txt.")

    # Calculate final unique IDs
    unique_counts = {label: len(ids) for label, ids in unique_counts.items()}
    
    stage_header.subheader("✅ Tracking Complete")
    prog_bar.empty() # Remove progress bar
    
    # Display final metrics dynamically
    with stats_placeholder.container():
        st.divider()
        st.subheader("📊 Lifetime Object Counts (Unique IDs)")
        if unique_counts:
            metric_cols = st.columns(len(unique_counts))
            for i, (label, count) in enumerate(unique_counts.items()):
                metric_cols[i].metric(label.capitalize() + "s", count)
        else:
            st.info("No tracked objects detected in this sequence.")

    # Provide the functioning download button
    with col_controls:
        with open(final_video_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Annotated Video", 
                data=f, 
                file_name="tracked_output.mp4",
                mime="video/mp4",
                type="primary"
            )
