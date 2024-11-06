# =========================
# Python 3.10.11
# =========================

import os
import cv2
import streamlit as st
import time
import math
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
            os.makedirs(uploads_dir, exist_ok=True)

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
                    os.makedirs(output_dir, exist_ok=True)
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
            os.makedirs(uploads_dir, exist_ok=True)
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
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, output_filename)

                    out = cv2.VideoWriter(
                        out_path,
                        cv2.VideoWriter_fourcc(*"h264"),
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

                    st.success("Video processing completed")

                    # Display the processed H.264 video
                    st.video(out_path)

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
        mode = st.selectbox("Select Mode", ["Photo", "Video", "Live"])

        if mode == "Photo":
            st.subheader("Capture Photo")
            st.text("Press 'Enable camera' to take a photo")
            # Capture photo using the camera input
            enable = st.checkbox("Enable camera", key="capture_photo_tab")
            picture = st.camera_input("Take a photo", disabled=not enable)

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
                st.success("Photo captured successfully")
                st.image(picture, caption="Captured Photo", use_column_width=True)

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

        elif mode == "Video":
            st.subheader("Record Video")
            st.text("Press 'Enable camera' to record a video")
            enable = st.checkbox("Enable camera", key="record_video_tab")
            start_recording = st.button(
                "Start Recording", disabled=not enable, help="Click to start recording"
            )

            # Initialize session state for recording
            if "start_recording" not in st.session_state:
                st.session_state["start_recording"] = False

            # Generate a unique filename once and store it in session state
            if "video_filename" not in st.session_state:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["video_filename"] = f"record_{timestamp}.mp4"

            video_filename = st.session_state["video_filename"]

            # Define directories for uploads and output
            uploads_dir = "uploads"
            output_dir = "output"
            os.makedirs(uploads_dir, exist_ok=True)
            os.makedirs(output_dir, exist_ok=True)

            # Paths for recorded and processed videos
            video_path = os.path.join(uploads_dir, video_filename)
            processed_video_path = os.path.join(output_dir, video_filename)

            video_placeholder = st.empty()

            # Start recording if the button is clicked
            if start_recording:
                st.session_state["start_recording"] = True

                # Initialize video capture and writer
                cap = cv2.VideoCapture(0)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                out = cv2.VideoWriter(
                    video_path,
                    cv2.VideoWriter_fourcc(*"h264"),
                    fps,
                    (frame_width, frame_height),
                )

                stop_recording = st.button(
                    "Stop Recording", help="Click to stop recording"
                )

                # Recording loop
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

                    # Stop recording when button is clicked
                    if stop_recording:
                        break

                # Release video capture and writer
                cap.release()
                out.release()
                st.session_state["start_recording"] = False

            if "video_recorded" not in st.session_state:
                st.session_state["video_recorded"] = False

            # Display the recorded video after completion
            if st.session_state["start_recording"] and os.path.exists(video_path):
                st.success("Video recording completed")

                with open(video_path, "rb") as f:
                    f.read()
                st.video(video_path)

                st.session_state["video_recorded"] = True

            # Start Detection button
            if st.session_state["video_recorded"] == True and st.button(
                "Start Detection", help="Click to start detection"
            ):
                # Process the recorded video
                with st.spinner("Processing detection..."):
                    # Initialize video capture and writer
                    cap = cv2.VideoCapture(video_path)
                    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    out = cv2.VideoWriter(
                        processed_video_path,
                        cv2.VideoWriter_fourcc(*"h264"),
                        fps,
                        (frame_width, frame_height),
                    )

                    frame_count = 0
                    progress_text = st.empty()
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1

                        # Update progress text
                        progress_text.warning(
                            f"Processing frame {frame_count}/{total_frames}"
                        )

                        processed_frame = process_frame(
                            frame
                        )  # Your detection function
                        out.write(processed_frame)

                    cap.release()
                    out.release()

                    st.success("Video processing completed")

                    st.video(processed_video_path)

                    os.remove(video_path)

                    # Provide download button for processed video
                    with open(processed_video_path, "rb") as f:
                        yolo_data = f.read()

                    @st.fragment
                    def downloadButton():
                        st.download_button(
                            label="Download Video",
                            data=yolo_data,
                            file_name=video_filename,
                            mime="video/mp4",
                            help="Click to download the processed video",
                        )
                        with st.spinner("Waiting for 3 seconds!"):
                            time.sleep(3)

                    downloadButton()

        elif mode == "Live":
            st.subheader("Live Detection")
            st.text("Press 'Start Webcam' to open the webcam")
            start_button = st.button(
                "Start Webcam", key="start_webcam", help="Click to start webcam"
            )
            stop_button_placeholder = st.empty()  # Initially hide the stop button

            # Placeholder for displaying frames
            frame_placeholder = st.empty()

            # Flag to track webcam state
            webcam_running = False

            if start_button:
                webcam_running = True

                # Code for webcam video capture
                cap = cv2.VideoCapture(0)
                cap.set(3, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                cap.set(4, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fps = int(cap.get(cv2.CAP_PROP_FPS))

                stop_button = stop_button_placeholder.button(
                    "Stop Webcam", key="stop_webcam", help="Click to stop webcam"
                )  # Show stop button only after start

                fps_text = st.empty()
                while webcam_running and cap.isOpened():
                    success, frame = cap.read()

                    if not success:
                        st.warning("Failed to capture frame from webcam")
                        break

                    # Process the frame using your YOLO model
                    frame = process_frame(frame)

                    # Convert BGR (OpenCV) to RGB for Streamlit display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Display the frame in the browser
                    frame_placeholder.image(
                        frame_rgb, channels="RGB", use_column_width=True
                    )

                    fps_text.warning(f"FPS: {fps:.2f}")

                    # Check if the stop button is pressed
                    if stop_button:
                        webcam_running = False
                        cap.release()

                        # Hide stop button and frame placeholder
                        stop_button_placeholder.empty()
                        frame_placeholder.empty()
                        break
