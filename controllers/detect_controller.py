import os
import cv2
import streamlit as st
import math
import datetime
import controllers.event_controller as ec
from ultralytics import YOLO

class_names = [
    "Others",
    "Plastic",
    "Straw",
    "Paper",
    "Tissue",
    "Bottle",
    "Beverage Carton Box",
    "Cigarette Pack",
    "Carton",
    "Food Container",
]
class_colors = {
    "Others": (255, 0, 0),
    "Plastic": (255, 0, 128),
    "Straw": (255, 0, 255),
    "Paper": (179, 0, 255),
    "Tissue": (0, 255, 0),
    "Bottle": (0, 255, 255),
    "Beverage Carton Box": (0, 128, 255),
    "Cigarette Pack": (0, 0, 255),
    "Carton": (255, 255, 0),
    "Food Container": (255, 128, 0),
}
model = YOLO("garbage.pt")


# Define function for detecting objects in a frame
def process_frame(frame):
    results = model(frame, stream=True)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            if cls < len(class_names):
                current_class = class_names[cls]
                detected_objects.append(current_class)
                color = class_colors[current_class]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{current_class} {conf}",
                    (max(0, x1), max(35, y1)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )

    # Save detected image and log to database if objects are detected
    if detected_objects:
        file_path = ec.save_detected_image(frame)
        ec.log_event_to_db(file_path, detected_objects)
    return frame


# =================================
# Static Image or Video Detection
# =================================
# Define function for uploaded image from file
def handle_uploaded_file(uploaded_file, file_type, process_func, mime_type):
    uploads_dir = os.path.join("uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Check if the uploaded_file has a 'name' attribute, otherwise create a temporary filename
    if hasattr(uploaded_file, "name"):
        file_name = uploaded_file.name
        file_data = uploaded_file.getbuffer()
        file_path = os.path.join(uploads_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(file_data)
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        file_name = f"capture_{timestamp}.jpg"
        file_path = os.path.join(uploads_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file)

    if file_type == "image":
        st.image(file_path, caption="Uploaded Image", use_column_width=True)
    elif file_type == "video":
        st.video(file_path)

    unique_key = f"{file_name}"
    if f"{unique_key}_clicked" not in st.session_state:
        st.session_state[f"{unique_key}_clicked"] = False

    if (
        st.button("Start Detection", key=unique_key)
        or st.session_state[f"{unique_key}_clicked"]
    ):
        st.session_state[f"{unique_key}_clicked"] = True
        with st.spinner("Processing detection..."):
            if file_type == "image" and isinstance(uploaded_file, bytes):
                processed_data = process_func(uploaded_file)
            else:
                processed_data = process_func(file_path)
            output_dir = "output"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)

            st.success(f"{file_type.capitalize()} processing completed")

            if file_type == "image":
                cv2.imwrite(output_path, processed_data)
                processed_data_rgb = cv2.cvtColor(processed_data, cv2.COLOR_BGR2RGB)
                st.image(processed_data_rgb, use_column_width=True)
            else:
                st.video(output_path)

            os.remove(file_path)

            with open(output_path, "rb") as f:
                yolo_data = f.read()

            @st.fragment
            def download_button():
                st.download_button(
                    "Download",
                    yolo_data,
                    file_name=file_name,
                    mime=mime_type,
                )

            download_button()


# Define function for uploaded video from file
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join("output", os.path.basename(video_path))
    out = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"H264"), fps, (frame_width, frame_height)
    )

    frames_processed = 0
    progress_text = st.empty()
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frames_processed += 1
        progress_text.warning(f"Processing frame {frames_processed}/{total_frames}")
        out.write(process_frame(frame))
    cap.release()


# =================================
# Dynamic Image or Video Detection
# =================================
# Define function for recorded video from camera
def record_video():
    start_recording = st.button("Start Recording")
    if "recording" not in st.session_state:
        st.session_state["recording"] = False

    if "video_filename" not in st.session_state:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        st.session_state["video_filename"] = f"record_{timestamp}.mp4"
    video_filename = st.session_state["video_filename"]

    uploads_dir = "uploads"
    output_dir = "output"
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, video_filename)
    processed_video_path = os.path.join(output_dir, video_filename)

    video_placeholder = st.empty()

    if start_recording:
        st.session_state["recording"] = True
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
        out = cv2.VideoWriter(
            video_path,
            cv2.VideoWriter_fourcc(*"H264"),
            fps,
            (frame_width, frame_height),
        )
        st.info("Recording video...")
        st.info("Press 'Stop Recording' to end recording")
        stop_recording = st.button("Stop Recording")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_recording:
                break
            out.write(frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        cap.release()
        out.release()
        st.session_state["recording"] = False

    if os.path.exists(video_path):
        st.success("Video recording completed")
        with open(video_path, "rb") as f:
            f.read()
        st.video(video_path)

        process_video(video_path)

        st.video(processed_video_path)

        os.remove(video_path)

        with open(processed_video_path, "rb") as f:
            yolo_data = f.read()

        @st.fragment
        def download_button():
            st.download_button(
                "Download",
                yolo_data,
                file_name=st.session_state.video_filename,
                mime="video/mp4",
            )

        download_button()


# Define function for live video detection from camera
def live_detection():
    cap = cv2.VideoCapture(0)
    cap.set(3, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.set(4, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    webcam_running = True
    frame_placeholder = st.empty()
    fps_text = st.empty()
    stop_button = st.button("Stop")
    while webcam_running and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        fps_text.warning(f"FPS: {fps:.2f}")
        if stop_button:
            webcam_running = False
            cap.release()
            stop_button.empty()
            frame_placeholder.empty()
            break
