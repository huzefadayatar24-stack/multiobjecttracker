import streamlit as st
import cv2
import tempfile
import os
import subprocess
import numpy as np
from PIL import Image
from ultralytics import YOLO
import supervision as sv

# ---------- PAGE SETUP & MODERN UI ----------
st.set_page_config(page_title="Vision MOT Dashboard", layout="wide", page_icon="🎯")

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
        max-height: 60vh; 
        width: auto !important; 
        max-width: 100%;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Vision MOT & Live Detection Engine</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Object Tracking & Real-Time Camera Inference</p>', unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()
all_classes = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---------- CONTROLS & LAYOUT ----------
col_controls, col_stage = st.columns([1, 2.5], gap="large")

with col_controls:
    st.subheader("⚙️ Settings")
    input_mode = st.radio("Select Input Mode", ["📁 Upload Video", "📷 Live Camera"], index=0)
    
    conf_threshold = st.slider("Confidence Threshold", 0.10, 0.90, 0.30, 0.05)
    iou_threshold = st.slider("IoU Threshold", 0.20, 0.80, 0.50, 0.05)
    
    selected_labels = st.multiselect(
        "Classes to Detect",
        options=list(all_classes.values()),
        default=list(all_classes.values())
    )
    selected_class_ids = [k for k, v in all_classes.items() if v in selected_labels]

    st.divider()
    
    uploaded_file = None
    camera_image = None
    
    if input_mode == "📁 Upload Video":
        uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
        start_btn = st.button("▶️ Process Video Tracking")
    else:
        st.info("Click 'Take Photo' below to run live detection through your camera.")
        camera_image = st.camera_input("Camera Viewfinder")

with col_stage:
    # -------------------------------------------------------------
    # MODE 1: LIVE CAMERA CAPTURE
    # -------------------------------------------------------------
    if input_mode == "📷 Live Camera":
        st.subheader("📷 Live Camera Detection")
        
        if camera_image is not None:
            # Convert the camera bytes to an OpenCV image
            bytes_data = camera_image.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Run YOLO inference
            results = model(cv2_img, conf=conf_threshold, iou=iou_threshold, classes=selected_class_ids, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
            
            labels = [f"{all_classes.get(cid, 'obj')} {conf:0.2f}" for cid, conf in zip(detections.class_id, detections.confidence)]
            
            annotated = box_annotator.annotate(scene=cv2_img.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            
            # Display the annotated frame
            st.image(annotated, channels="BGR", use_container_width=True)
            
            # Display stats
            st.divider()
            st.subheader("📊 Detected Objects")
            counts = {}
            for cid in detections.class_id:
                name = all_classes.get(cid, "obj")
                counts[name] = counts.get(name, 0) + 1
                
            if counts:
                cols = st.columns(len(counts))
                for idx, (label, count) in enumerate(counts.items()):
                    cols[idx].metric(label.capitalize() + "s", count)
            else:
                st.info("No objects detected above the selected confidence threshold.")
        else:
            st.info("Awaiting camera input. Allow browser camera permissions if prompted.")

    # -------------------------------------------------------------
    # MODE 2: VIDEO PROCESSING & TRACKING
    # -------------------------------------------------------------
    elif input_mode == "📁 Upload Video":
        stage_header = st.empty()
        status_text = st.empty()
        progress_bar = st.empty()
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        if not uploaded_file:
            stage_header.subheader("Stage: Standby")
            status_text.info("Upload a video and click Process.")

        if uploaded_file is not None and start_btn:
            input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            input_temp.write(uploaded_file.read())
            input_path = input_temp.name

            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            max_width = 640
            if width > max_width:
                scale = max_width / width
                width, height = int(width * scale), int(height * scale)

            output_path = os.path.join(tempfile.gettempdir(), "raw_hd_output.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            tracker = sv.ByteTrack(track_activation_threshold=conf_threshold, lost_track_buffer=45)
            box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_scale=0.45, text_thickness=1)
            
            unique_counts = {}

            stage_header.subheader("⚙️ Tracking Active...")
            status_text.info("Processing frames. Please wait...")
            prog_bar = progress_bar.progress(0)

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (width, height))
                results = model(frame, conf=conf_threshold, iou=iou_threshold, classes=selected_class_ids, imgsz=640, verbose=False)[0]
                
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

                frame_idx += 1
                if total_frames > 0 and frame_idx % 5 == 0:
                    prog_bar.progress(min(frame_idx / total_frames, 1.0))

            cap.release()
            writer.release()
            
            stage_header.subheader("Finalizing Video...")
            status_text.info("Generating browser-ready MP4...")
            prog_bar.progress(1.0)
            
            h264_output_path = os.path.join(tempfile.gettempdir(), "h264_hd_output.mp4")
            try:
                subprocess.run([
                    "ffmpeg", "-y", "-i", output_path,
                    "-vcodec", "libx264", "-crf", "18", "-preset", "medium",
                    "-pix_fmt", "yuv420p", h264_output_path
                ], check=True, capture_output=True, text=True)
                final_video_path = h264_output_path
            except Exception as e:
                final_video_path = output_path

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
                    st.info("No tracked objects detected.")

            with col_controls:
                with open(final_video_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Annotated Video", 
                        data=f, 
                        file_name="tracked_output.mp4",
                        mime="video/mp4",
                        type="primary"
                    )
