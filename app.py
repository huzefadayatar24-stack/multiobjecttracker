import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import supervision as sv

# ---------- PAGE SETUP ----------
st.set_page_config(page_title="Real-Time MOT (YOLOv8 + ByteTrack)", layout="wide")
st.title("🚦 Real-Time Multi-Object Tracking")
st.write("Upload a video. The app will detect and track objects (Person, Car, Bus, Truck) and give you a downloadable annotated video.")

# ---------- LOAD MODEL (cached so it only loads once) ----------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")  # nano model = fastest, best for free CPU servers

model = load_model()

# ---------- SIDEBAR SETTINGS ----------
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.3, 0.05)
iou_threshold = st.sidebar.slider("IoU threshold", 0.1, 1.0, 0.5, 0.05)

all_classes = {0: "person", 2: "car", 5: "bus", 7: "truck"}
selected_labels = st.sidebar.multiselect(
    "Classes to track",
    options=list(all_classes.values()),
    default=list(all_classes.values())
)
selected_class_ids = [k for k, v in all_classes.items() if v in selected_labels]

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:

    # Save uploaded file to a temp location so OpenCV can read it
    input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    input_temp.write(uploaded_file.read())
    input_path = input_temp.name

    if st.button("▶️ Start Processing"):

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Downscale large videos so free CPU server can keep up
        max_width = 640
        if width > max_width:
            scale = max_width / width
            width, height = int(width * scale), int(height * scale)

        output_path = os.path.join(tempfile.gettempdir(), "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracker = sv.ByteTrack()
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        progress_bar = st.progress(0, text="Processing video... please wait")
        preview_placeholder = st.empty()

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (width, height))

            results = model(frame, conf=conf_threshold, iou=iou_threshold,
                             classes=selected_class_ids, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            labels = [
                f"#{tracker_id} {all_classes.get(class_id, 'obj')}"
                for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
            ]

            annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

            writer.write(annotated)

            # Show a live-ish preview every 15 frames (not every frame — keeps it fast)
            if frame_idx % 15 == 0:
                preview_placeholder.image(annotated, channels="BGR",
                                           caption="Live preview (updates periodically)",
                                           use_container_width=True)

            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0),
                                       text=f"Processing frame {frame_idx}/{total_frames}")

        cap.release()
        writer.release()
        progress_bar.progress(1.0, text="Done!")

        st.success("✅ Processing complete!")
        st.video(output_path)

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download annotated video", f, file_name="tracked_output.mp4")
