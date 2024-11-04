# =========================
# Python 3.10.11
# =========================

import os
import cv2
import streamlit as st
import time
import math
from ultralytics import YOLO

# Constants
frame_width = 1280
frame_height = 720
font = cv2.FONT_HERSHEY_SIMPLEX

# Streamlit UI
st.title("Application for Detecting Littering Actions using YOLO")

st.sidebar.title("Dashboard")

# Sidebar options for input source
option = st.sidebar.selectbox(
    "Choose an option",
    [
        "Detect from Image File",
        "Detect from Video File",
        "Open Webcam",
    ],
)

model = YOLO("garbage.pt")

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


def processFrame(frame):
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


if option == "Detect from Image File":
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

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

        if st.button("Start Detection"):
            # Show spinner and warning message
            with st.spinner("Processing detection..."):
                # Process image
                frame = cv2.imread(image_path)
                frame_width = frame.shape[1]
                frame_height = frame.shape[0]

                processed_frame = processFrame(frame)

                # Save and display processed image
                output_dir = "output"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_image_path = os.path.join(output_dir, uploaded_image.name)

                cv2.imwrite(output_image_path, processed_frame)

                st.success("Image processing completed")

                # Convert BGR to RGB
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                st.image(
                    processed_frame_rgb,
                    caption="Processed Image",
                    use_column_width=True,
                )

                with open(output_image_path, "rb") as f:
                    yolo_data = f.read()

                # Optionally, remove the temporary uploaded file
                os.remove(image_path)

                @st.experimental_fragment
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

elif option == "Detect from Video File":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_video is not None:
        uploads_dir = os.path.join("uploads")
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        video_path = os.path.join(uploads_dir, uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.video(video_path)

        if st.button("Start Detection"):
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

                    frame = processFrame(frame)
                    out.write(frame)

                cap.release()
                out.release()

                st.success("Video processing completed")

                with open(out_path, "rb") as f:
                    yolo_data = f.read()

                # Optionally, remove the temporary uploaded file
                os.remove(video_path)

                @st.experimental_fragment
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

elif option == "Open Webcam":
    st.text("Press 'Start Webcam' to open the webcam")
    start_button = st.button("Start Webcam")
    stop_button = st.empty()  # Initially hide the stop button

    # Placeholder for displaying frames
    frame_placeholder = st.empty()

    # Flag to track webcam state
    webcam_running = False

    if start_button:
        webcam_running = True

        cap = cv2.VideoCapture(0)
        cap.set(3, frame_width)
        cap.set(4, frame_height)

        stop_button = st.button("Stop Webcam")  # Show stop button only after start

    while webcam_running and cap.isOpened():
        success, frame = cap.read()

        if not success:
            st.warning("Failed to capture frame from webcam.")
            break

        # Process the frame using your YOLO model
        frame = processFrame(frame)

        # Convert BGR (OpenCV) to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in the browser
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Check if the stop button is pressed
        if stop_button:
            webcam_running = False
            cap.release()

            # Hide stop button and frame placeholder
            stop_button.text = ""
            frame_placeholder.empty()
            break
