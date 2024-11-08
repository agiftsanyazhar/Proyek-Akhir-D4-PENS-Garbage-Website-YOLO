import cv2
import streamlit as st
import numpy as np
import controllers.detect_controller as dc


def app():
    st.title("Application for Detecting Littering Actions using YOLO - Detect")

    detect_image_tab, detect_video_tab, detect_webcam_tab = st.tabs(
        ["Detect from Image File", "Detect from Video File", "Open Webcam"]
    )

    with detect_image_tab:
        uploaded_image = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"]
        )
        if uploaded_image:
            process_func = lambda path: dc.process_frame(cv2.imread(path))
            dc.handle_uploaded_file(uploaded_image, "image", process_func, "image/jpeg")

    with detect_video_tab:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
        if uploaded_video:
            process_func = lambda path: dc.process_video(path)
            dc.handle_uploaded_file(uploaded_video, "video", process_func, "video/mp4")

    with detect_webcam_tab:
        mode = st.selectbox("Select Mode", ["Photo", "Video", "Live"])

        if mode == "Photo":
            st.subheader("Capture Photo")
            enable = st.checkbox("Enable camera", key="capture_photo_tab")
            picture = st.camera_input("Take a photo", disabled=not enable)
            if picture:
                image_data = picture.getvalue()
                process_func = lambda data: dc.process_frame(
                    cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                )
                dc.handle_uploaded_file(image_data, "image", process_func, "image/jpeg")

        elif mode == "Video":
            st.subheader("Record Video")
            enable = st.checkbox("Enable camera", key="record_video_tab")
            if enable:
                dc.record_video()

        elif mode == "Live":
            st.subheader("Live Detection")
            if st.button("Start Webcam", key="start_webcam"):
                dc.live_detection()
