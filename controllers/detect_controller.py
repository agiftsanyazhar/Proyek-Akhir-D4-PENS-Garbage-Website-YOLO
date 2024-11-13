import os
import cv2
import streamlit as st
import math
import datetime
import time
import ffmpeg
import re
import random
from ultralytics import YOLO
from controllers.event_controller import save_detected_image, store


# Automatically generate colors for classes from the model
def generate_class_colors(model):
    if hasattr(model, "names") and isinstance(model.names, list):
        num_classes = len(model.names)
    elif hasattr(model, "names") and isinstance(model.names, dict):
        num_classes = len(model.names)
    else:
        raise ValueError("The model does not have class names defined.")

    # Generate random distinct colors for each class
    random.seed(42)  # For consistent color generation
    colors = {
        cls_name: (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        for cls_id, cls_name in model.names.items()
    }

    return colors


model = YOLO("models/garbage.pt")
# model = YOLO("models/train12.pt")

class_colors = generate_class_colors(model)


# Function to get the list of available cameras dynamically
def get_camera_list():
    available_cameras = []
    index = 0

    while True:
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            available_cameras.append((index, f"Camera {index}"))
            cap.release()
            index += 1
        else:
            break

    return available_cameras


def camera_preview(type=None):
    camera_list = get_camera_list()
    if not camera_list:
        st.error("No camera devices found")
        return

    selected_camera = st.selectbox(
        "Select Camera", camera_list, format_func=lambda x: x[1]
    )

    preview_placeholder = st.empty()
    preview_button = st.button("Preview")

    if preview_button:
        st.session_state["preview_running"] = True
        st.session_state["start_detection"] = False

    if "preview_running" in st.session_state and st.session_state["preview_running"]:
        start_button_placeholder = st.empty()
        start_button = (
            start_button_placeholder.button("Start Detection")
            if type == "live"
            else start_button_placeholder.button("Stop Preview")
        )

        cap = cv2.VideoCapture(selected_camera[0])
        preview_running = True

        while preview_running and cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                st.error("Failed to load the camera")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preview_placeholder.image(
                frame_rgb,
                caption="Camera Preview",
                channels="RGB",
                use_column_width=True,
            )

            if start_button:
                start_button_placeholder.empty()
                st.session_state["preview_running"] = False
                st.session_state["start_detection"] = True
                break

        cap.release()

    return selected_camera


def check_cctv_connection(url):
    start_time = time.time()
    while time.time() - start_time < 30:
        try:
            ffmpeg.probe(url)
            st.success("CCTV connected successfully")
            return
        except ffmpeg.Error as e:
            error_msg = e.stderr.decode()
            match = re.search(r"method DESCRIBE failed: \d{3} (.+)", error_msg)
            if match:
                short_error_msg = match.group(1)
            else:
                short_error_msg = "Please check the URL or credentials"
            st.error(f"Failed to connect to CCTV: {short_error_msg}")
            time.sleep(5)
    st.error("Connection attempt timed out after 30 seconds")


# Define function for detecting objects in a frame
def process_frame(frame):
    results = model(frame, stream=True)
    detected_objects = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])

            if cls < len(model.names):
                current_class = model.names[cls]
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

    if detected_objects:
        file_path = save_detected_image(frame)
        store(file_path, detected_objects)

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

            output_dir = "outputs"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, file_name)

            st.success(f"{file_type.capitalize()} processing completed")
            if file_type == "image":
                cv2.imwrite(output_path, processed_data)
                processed_data_rgb = cv2.cvtColor(processed_data, cv2.COLOR_BGR2RGB)
                st.image(
                    processed_data_rgb, caption="Processed Image", use_column_width=True
                )
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

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join("outputs", os.path.basename(video_path))
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
        progress_text.info(f"Processing frame {frames_processed}/{total_frames}")
        out.write(process_frame(frame))

    cap.release()


# =================================
# Dynamic Image or Video Detection
# =================================
# Define function for recorded video from camera
def record_video():
    selected_camera = camera_preview()

    if "video_filename" not in st.session_state:
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        st.session_state["video_filename"] = f"record_{timestamp}.mp4"
    video_filename = st.session_state["video_filename"]

    uploads_dir = "uploads"
    output_dir = "outputs"
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(uploads_dir, video_filename)
    processed_video_path = os.path.join(output_dir, video_filename)

    if "start_detection" in st.session_state and st.session_state["start_detection"]:
        start_recording = st.button("Start Recording")

        video_placeholder = st.empty()
        if start_recording:
            cap = cv2.VideoCapture(selected_camera[0])
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            out = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"H264"),
                fps,
                (frame_width, frame_height),
            )
            st.info("Recording video... Press 'Stop Recording' to end recording")

            stop_recording = st.button("Stop Recording")

            while cap.isOpened():
                ret, frame = cap.read()

                if not ret or stop_recording:
                    stop_recording.empty()
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    frame_rgb, channels="RGB", use_column_width=True
                )
                out.write(frame)

            cap.release()
            out.release()

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
                    file_name=st.session_state["video_filename"],
                    mime="video/mp4",
                )

            download_button()


# Define function for live video detection from camera
def live_detection():
    selected_camera = camera_preview(type="live")

    if "start_detection" in st.session_state and st.session_state["start_detection"]:
        info_message = st.info("Live detection started")

        cap = cv2.VideoCapture(selected_camera[0])
        cap.set(3, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        cap.set(4, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        pTime = 0
        webcam_running = True

        frame_placeholder = st.empty()
        fps_text = st.empty()
        stop_button_placeholder = st.empty()
        stop_button = stop_button_placeholder.button("Stop Detection")

        while webcam_running and cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame = process_frame(frame)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(
                frame_rgb, caption="Live Feed", channels="RGB", use_column_width=True
            )
            fps_text.info(f"FPS: {fps:.2f}")

            if stop_button:
                webcam_running = False
                st.session_state["start_detection"] = False
                info_message.empty()
                stop_button_placeholder.empty()
                frame_placeholder.empty()
                fps_text.empty()
                break
        cap.release()


def cctv_detection(url):
    info_message = st.info("Connecting to CCTV...")

    check_cctv_connection(url)

    cap = cv2.VideoCapture(url)
    cap.set(3, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.set(4, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    pTime = 0
    webcam_running = True

    frame_placeholder = st.empty()
    fps_text = st.empty()
    stop_button_placeholder = st.empty()
    stop_button = stop_button_placeholder.button("Stop CCTV")

    while webcam_running and cap.isOpened():
        success, frame = cap.read()

        if not success:
            st.error("Lost connection to the CCTV feed")
            break

        frame = process_frame(frame)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(
            frame_rgb, caption="Live Feed", channels="RGB", use_column_width=True
        )
        fps_text.info(f"FPS: {fps:.2f}")

        if stop_button:
            webcam_running = False
            st.session_state["start_detection"] = False
            info_message.empty()
            stop_button_placeholder.empty()
            frame_placeholder.empty()
            fps_text.empty()
            break
    cap.release()
