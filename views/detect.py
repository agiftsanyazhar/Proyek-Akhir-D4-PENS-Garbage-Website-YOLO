# =========================
# Python 3.10.11
# =========================

import os
import cv2
import streamlit as st
import time
import math
import subprocess
from ultralytics import YOLO
import datetime
import numpy as np


def app():
    # Define class names
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

    # Define colors for each class (adjust as needed)
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

    # Main page title
    st.title("Application for Detecting Littering Actions using YOLO - Detect")

    model = YOLO("garbage.pt")

    def process_frame(frame):
        # Perform object detection
        results = model(frame, stream=True)

        # Process each result
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Extract box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])

                # Ensure the class index is within the range of classNames
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
                else:
                    print(f"Warning: Class index {cls} is out of range for classNames")

        return frame

    detect_image_tab, detect_video_tab, detect_webcam_tab = st.tabs(
        ["Detect from Image File", "Detect from Video File", "Open Webcam"]
    )

    with detect_image_tab:
        uploaded_image = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_image is not None:
            # Save uploaded image
            uploads_dir = os.path.join("uploads")
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)

            image_path = os.path.join(uploads_dir, uploaded_image.name)

            # Display uploaded image
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            st.image(image_path, caption="Uploaded Image", use_column_width=True)

            if st.button(
                "Start Detection",
                help="Click to start detection",
                key=detect_image_tab,
            ):
                # Show spinner and warning message
                with st.spinner("Processing detection..."):
                    # Process image
                    frame = cv2.imread(image_path)
                    frame_width = frame.shape[1]
                    frame_height = frame.shape[0]

                    processed_frame = process_frame(frame)

                    # Save and display processed image
                    output_dir = "output"
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    output_image_path = os.path.join(output_dir, uploaded_image.name)

                    cv2.imwrite(output_image_path, processed_frame)

                    st.success("Image processing completed")

                    # Convert BGR to RGB
                    processed_frame_rgb = cv2.cvtColor(
                        processed_frame, cv2.COLOR_BGR2RGB
                    )

                    st.image(
                        processed_frame_rgb,
                        caption="Processed Image",
                        use_column_width=True,
                    )

                    # Optionally, remove the temporary uploaded file
                    os.remove(image_path)

                    with open(output_image_path, "rb") as f:
                        yolo_data = f.read()

                    @st.fragment
                    def downloadButton():
                        st.download_button(
                            label="Download Image",
                            data=yolo_data,
                            file_name=f"{uploaded_image.name}",
                            mime="image/jpeg",
                            help="Click to download the processed image",
                        )

                        with st.spinner("Waiting for 3 seconds!"):
                            time.sleep(3)

                    downloadButton()

    with detect_video_tab:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

        if uploaded_video is not None:
            uploads_dir = os.path.join("uploads")
            if not os.path.exists(uploads_dir):
                os.makedirs(uploads_dir)
            video_path = os.path.join(uploads_dir, uploaded_video.name)

            with open(video_path, "wb") as f:
                f.write(uploaded_video.getbuffer())
            st.video(video_path)

            if st.button(
                "Start Detection",
                help="Click to start detection",
                key=detect_video_tab,
            ):
                with st.spinner("Processing detection..."):
                    cap = cv2.VideoCapture(video_path)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    output_filename = os.path.splitext(uploaded_video.name)[0] + ".mp4"
                    out_dir = "output"
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                    out_path = os.path.join(out_dir, output_filename)

                    out = cv2.VideoWriter(
                        out_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (frame_width, frame_height),
                    )

                    frame_count = 0
                    progress_text = st.empty()  # Placeholder for progress text
                    while cap.isOpened():
                        success, frame = cap.read()

                        if not success:
                            break

                        frame_count += 1

                        # Update progress text
                        progress_text.warning(
                            f"Processing frame {frame_count}/{total_frames}"
                        )

                        frame = process_frame(frame)
                        out.write(frame)

                    cap.release()
                    out.release()

                    # Convert to H.264 codec using ffmpeg
                    h264_output_filename = (
                        os.path.splitext(uploaded_video.name)[0] + "_h264.mp4"
                    )
                    h264_out_path = os.path.join(out_dir, h264_output_filename)

                    # Run ffmpeg to re-encode
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-i",
                            out_path,
                            "-vcodec",
                            "libx264",
                            "-y",
                            h264_out_path,
                        ]
                    )

                    st.success("Video processing completed")

                    # Display the processed H.264 video
                    st.video(h264_out_path)

                    # Clean up the temporary files
                    os.remove(video_path)

                    with open(out_path, "rb") as f:
                        yolo_data = f.read()

                    @st.fragment
                    def downloadButton():
                        st.download_button(
                            label="Download Video",
                            data=yolo_data,
                            file_name=output_filename,
                            mime="video/mp4",
                            help="Click to download the video",
                        )
                        with st.spinner("Waiting for 3 seconds!"):
                            time.sleep(3)

                    downloadButton()

    with detect_webcam_tab:
        # Dropdown for selecting mode
        mode = st.selectbox("Select Mode", ["Photo", "Video"])

        if mode == "Photo":
            # Capture photo using the camera input
            enable = st.checkbox("Enable camera")
            picture = st.camera_input("Take a picture", disabled=not enable)

            if picture:
                # Generate a unique filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                capture_filename = f"capture_{timestamp}.jpg"

                # Define directories for uploads and output
                output_dir = "output"

                # Ensure the directories exist
                os.makedirs(output_dir, exist_ok=True)

                # Paths for captured and processed images
                processed_image_path = os.path.join(output_dir, capture_filename)

                # Display the captured image
                st.image(picture, caption="Captured Image", use_column_width=True)

                if picture is not None:
                    bytes_data = picture.getvalue()

                if st.button(
                    "Start Detection",
                    help="Click to start detection",
                    key=detect_image_tab,
                ):
                    # Process the captured image
                    with st.spinner("Processing detection..."):
                        frame = cv2.imdecode(
                            np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
                        )
                        processed_frame = process_frame(frame)

                        # Save the processed image to the output folder
                        cv2.imwrite(processed_image_path, processed_frame)

                        st.success("Image processing completed")

                        # Display the processed image
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

                        @st.fragment
                        def downloadButton():
                            st.download_button(
                                label="Download Image",
                                data=yolo_data,
                                file_name=capture_filename,
                                mime="image/jpeg",
                                help="Click to download the processed image",
                            )

                            with st.spinner("Waiting for 3 seconds!"):
                                time.sleep(3)

                        downloadButton()

        # else:
        # st.text("Press 'Start Webcam' to open the webcam")
        # start_button = st.button("Start Webcam", key="start_webcam")
        # stop_button_placeholder = st.empty()  # Initially hide the stop button

        # # Placeholder for displaying frames
        # frame_placeholder = st.empty()

        # if start_button:
        #     # Code for webcam video capture
        #     webcam_running = True
        #     cap = cv2.VideoCapture(0)
        #     cap.set(3, frame_width)
        #     cap.set(4, frame_height)

        #     stop_button = stop_button_placeholder.button(
        #         "Stop Webcam", key="stop_webcam"
        #     )  # Show stop button only after start

        #     while webcam_running and cap.isOpened():
        #         success, frame = cap.read()

        #         if not success:
        #             st.warning("Failed to capture frame from webcam.")
        #             break

        #         # Process the frame using your YOLO model
        #         frame = process_frame(frame)

        #         # Convert BGR (OpenCV) to RGB for Streamlit display
        #         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #         # Display the frame in the browser
        #         frame_placeholder.image(
        #             frame_rgb, channels="RGB", use_column_width=True
        #         )

        #         # Check if the stop button is pressed
        #         if stop_button:
        #             webcam_running = False
        #             cap.release()

        #             # Hide stop button and frame placeholder
        #             stop_button_placeholder.empty()
        #             frame_placeholder.empty()
        #             break
