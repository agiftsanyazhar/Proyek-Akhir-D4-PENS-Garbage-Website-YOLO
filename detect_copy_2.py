import os
import cv2
import streamlit as st
import time
import math
from ultralytics import YOLO
import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor


def app():
    # Define class names and colors
    classNames = [
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
        "Paper": (179, 0, 255),
        "Straw": (255, 0, 255),
        "Tissue": (0, 255, 0),
        "Bottle": (0, 255, 255),
        "Beverage Carton Box": (0, 128, 255),
        "Cigarette Pack": (0, 0, 255),
        "Carton": (255, 255, 0),
        "Food Container": (255, 128, 0),
    }

    st.title("Application for Detecting Littering Actions using YOLO - Detect")
    model = YOLO("garbage.pt")

    def process_frame(frame):
        results = model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                if cls < len(classNames):
                    currentClass = classNames[cls]
                    color = class_colors[currentClass]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{currentClass} {conf}"
                    cv2.putText(
                        frame,
                        text,
                        (max(0, x1), max(35, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                    )
        return frame

    def save_and_display_image(image_path, processed_frame):
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        output_image_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_image_path, processed_frame)
        st.success("Image processing completed")
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        st.image(processed_frame_rgb, caption="Processed Image", use_column_width=True)
        with open(output_image_path, "rb") as f:
            yolo_data = f.read()
        st.download_button(
            label="Download Image",
            data=yolo_data,
            file_name=os.path.basename(image_path),
            mime="image/jpeg",
        )

    def process_video(video_path):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_filename = os.path.splitext(os.path.basename(video_path))[0] + ".mp4"
        out_dir = "output"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, output_filename)
        out = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"H264"), fps, (frame_width, frame_height)
        )
        frame_count = 0
        progress_text = st.empty()
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame_count += 1
            progress_text.warning(f"Processing frame {frame_count}/{total_frames}")
            frame = process_frame(frame)
            out.write(frame)
        cap.release()
        out.release()
        st.success("Video processing completed")
        st.video(out_path)
        with open(out_path, "rb") as f:
            yolo_data = f.read()

        @st.fragment
        def download_button():
            st.download_button(
                label="Download Video",
                data=yolo_data,
                file_name=output_filename,
                mime="video/mp4",
            )

        download_button()

    detect_image_tab, detect_video_tab, detect_webcam_tab = st.tabs(
        ["Detect from Image File", "Detect from Video File", "Open Webcam"]
    )

    with detect_image_tab:
        uploaded_image = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_image is not None:
            uploads_dir = os.path.join("uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            image_path = os.path.join(uploads_dir, uploaded_image.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.image(image_path, caption="Uploaded Image", use_column_width=True)
            if st.button(
                "Start Detection", help="Click to start detection", key=detect_image_tab
            ):
                with st.spinner("Processing detection..."):
                    frame = cv2.imread(image_path)
                    processed_frame = process_frame(frame)
                    save_and_display_image(image_path, processed_frame)
                    os.remove(image_path)

    with detect_video_tab:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
        if uploaded_video is not None:
            uploads_dir = os.path.join("uploads")
            os.makedirs(uploads_dir, exist_ok=True)
            video_path = os.path.join(uploads_dir, uploaded_video.name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            st.video(video_path)
            if st.button(
                "Start Detection", help="Click to start detection", key=detect_video_tab
            ):
                with st.spinner("Processing detection..."):
                    process_video(video_path)
                    os.remove(video_path)

    with detect_webcam_tab:
        mode = st.selectbox("Select Mode", ["Photo", "Video", "Live"])
        if mode == "Photo":
            st.subheader("Capture Photo")
            st.text("Press 'Enable camera' to take a photo")
            enable = st.checkbox("Enable camera", key="capture_photo_tab")
            picture = st.camera_input("Take a photo", disabled=not enable)
            if picture:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                capture_filename = f"capture_{timestamp}.jpg"
                output_dir = "output"
                os.makedirs(output_dir, exist_ok=True)
                processed_image_path = os.path.join(output_dir, capture_filename)
                st.success("Photo captured successfully")
                st.image(picture, caption="Captured Photo", use_column_width=True)
                if picture is not None:
                    bytes_data = picture.getvalue()
                if st.button(
                    "Start Detection",
                    help="Click to start detection",
                    key=detect_image_tab,
                ):
                    with st.spinner("Processing detection..."):
                        frame = cv2.imdecode(
                            np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
                        )
                        processed_frame = process_frame(frame)
                        cv2.imwrite(processed_image_path, processed_frame)
                        st.success("Image processing completed")
                        processed_frame_rgb = cv2.cvtColor(
                            processed_frame, cv2.COLOR_BGR2RGB
                        )
                        st.image(
                            processed_frame_rgb,
                            caption="Processed Image",
                            use_column_width=True,
                        )
                        with open(processed_image_path, "rb") as f:
                            yolo_data = f.read()
                        st.download_button(
                            label="Download Image",
                            data=yolo_data,
                            file_name=capture_filename,
                            mime="image/jpeg",
                        )

        elif mode == "Video":
            st.subheader("Record Video")
            st.text("Press 'Enable camera' to record a video")
            enable = st.checkbox("Enable camera", key="record_video_tab")
            start_recording = st.button(
                "Start Recording", disabled=not enable, help="Click to start recording"
            )
            if "start_recording" not in st.session_state:
                st.session_state["start_recording"] = False
            if "video_filename" not in st.session_state:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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
                st.session_state["start_recording"] = True
                cap = cv2.VideoCapture(0)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                out = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"H264"),
                    fps,
                    (frame_width, frame_height),
                )
                stop_recording = st.button(
                    "Stop Recording", help="Click to stop recording"
                )
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        out.write(frame)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(
                            frame_rgb, channels="RGB", use_column_width=True
                        )
                    else:
                        break
                    if stop_recording:
                        break
                cap.release()
                out.release()
                st.session_state["start_recording"] = False
            if "video_recorded" not in st.session_state:
                st.session_state["video_recorded"] = False
            if st.session_state["start_recording"] and os.path.exists(video_path):
                st.success("Video recording completed")
                with open(video_path, "rb") as f:
                    f.read()
                st.video(video_path)
                st.session_state["video_recorded"] = True
            if st.session_state["video_recorded"] == True and st.button(
                "Start Detection", help="Click to start detection"
            ):
                with st.spinner("Processing detection..."):
                    process_video(video_path)
                    os.remove(video_path)

        elif mode == "Live":
            st.subheader("Live Detection")
            st.text("Press 'Start Webcam' to open the webcam")
            start_button = st.button(
                "Start Webcam", key="start_webcam", help="Click to start webcam"
            )
            stop_button_placeholder = st.empty()
            frame_placeholder = st.empty()
            webcam_running = False
            if start_button:
                webcam_running = True
                cap = cv2.VideoCapture(0)
                cap.set(3, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                cap.set(4, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                stop_button = stop_button_placeholder.button(
                    "Stop Webcam", key="stop_webcam", help="Click to stop webcam"
                )
                fps_text = st.empty()
                while webcam_running and cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        st.warning("Failed to capture frame from webcam")
                        break
                    frame = process_frame(frame)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        frame_rgb, channels="RGB", use_column_width=True
                    )
                    fps_text.warning(f"FPS: {fps:.2f}")
                    if stop_button:
                        webcam_running = False
                        cap.release()
                        stop_button_placeholder.empty()
                        frame_placeholder.empty()
                        break
