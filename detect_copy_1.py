import os
import cv2
import streamlit as st
import math
from ultralytics import YOLO
import datetime
import numpy as np


def app():
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

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                if cls < len(classNames):
                    currentClass = classNames[cls]
                    color = class_colors[currentClass]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{currentClass} {conf}",
                        (max(0, x1), max(35, y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2,
                    )
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
            file_name = f"capture_{datetime.datetime.now().strftime('%Y%m%d')}.jpg"
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
            and st.session_state[f"{unique_key}_clicked"]
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

                if file_type == "image":
                    cv2.imwrite(output_path, processed_data)
                    processed_data_rgb = cv2.cvtColor(processed_data, cv2.COLOR_BGR2RGB)

                st.success(f"{file_type.capitalize()} processing completed")

                if file_type == "image":
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
        out.release()

        return out_path

    # =================================
    # Dynamic Image or Video Detection
    # =================================
    # Define function for recorded video from camera
    def record_video():
        start_recording = st.button("Start Recording", disabled=not enable)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate a unique filename once and store it in session state
        if "video_filename" not in st.session_state:
            st.session_state["video_filename"] = f"record_{timestamp}.mp4"

        video_filename = st.session_state["video_filename"]
        video_path = os.path.join("uploads", video_filename)
        out_path = os.path.join("output", os.path.basename(video_path))

        if start_recording:
            cap = cv2.VideoCapture(0)
            frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            )
            fps = (
                int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
            )

            out = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*"H264"),
                fps,
                (frame_width, frame_height),
            )

            video_placeholder = st.empty()
            stop_recording = st.button("Stop Recording")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or stop_recording:
                    break

                out.write(frame)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(
                    frame_rgb, channels="RGB", use_column_width=True
                )

            cap.release()
            out.release()

        if os.path.exists(video_path):
            st.success("Video recording completed")
            st.video(video_path)
            process_video(video_path)

            st.video(out_path)

            os.remove(video_path)

            with open(out_path, "rb") as f:
                yolo_data = f.read()

            @st.fragment
            def download_button():
                st.download_button(
                    "Download",
                    yolo_data,
                    file_name=video_filename,
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

    st.title("Application for Detecting Littering Actions using YOLO - Detect")

    detect_image_tab, detect_video_tab, detect_webcam_tab = st.tabs(
        ["Detect from Image File", "Detect from Video File", "Open Webcam"]
    )

    with detect_image_tab:
        uploaded_image = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_image:
            handle_uploaded_file(
                uploaded_image,
                "image",
                lambda path: process_frame(cv2.imread(path)),
                "image/jpeg",
            )

    with detect_video_tab:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
        if uploaded_video:
            handle_uploaded_file(
                uploaded_video, "video", lambda path: process_video(path), "video/mp4"
            )

    with detect_webcam_tab:
        mode = st.selectbox("Select Mode", ["Photo", "Video", "Live"])
        if mode == "Photo":
            st.subheader("Capture Photo")
            enable = st.checkbox("Enable camera", key="capture_photo_tab")
            picture = st.camera_input("Take a photo", disabled=not enable)
            if picture:
                image_data = picture.getvalue()
                process_func = lambda data: process_frame(
                    cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                )
                handle_uploaded_file(image_data, "image", process_func, "image/jpeg")

        elif mode == "Video":
            st.subheader("Record Video")
            enable = st.checkbox("Enable camera", key="record_video_tab")
            record_video()

        elif mode == "Live":
            st.subheader("Live Detection")
            if st.button("Start Webcam", key="start_webcam"):
                live_detection()
