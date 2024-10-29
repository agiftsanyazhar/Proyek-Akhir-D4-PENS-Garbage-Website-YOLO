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
classNames = ["Others", "Plastic", "Straw", "Paper"]

# Define colors for each class (adjust as needed)
class_colors = {
    "Others": (255, 0, 0),
    "Plastic": (255, 0, 128),
    "Straw": (255, 0, 255),
    "Paper": (179, 0, 255),
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
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        # Display uploaded image
        st.image(image_path, caption="Uploaded Image", use_column_width=True)

        if st.button("Start Detection"):
            # Process image
            frame = cv2.imread(image_path)
            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            st.warning("Processing detection...")
            processed_frame = processFrame(frame)

            # Save and display processed image
            output_image_path = os.path.join("output", f"{uploaded_image.name}")
            cv2.imwrite(output_image_path, processed_frame)

            st.success("Image processing completed")

            # Convert BGR to RGB
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            st.image(
                processed_frame_rgb, caption="Processed Image", use_column_width=True
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

                with st.spinner("Waiting for 5 seconds!"):
                    time.sleep(5)

            downloadButton()

elif option == "Detect from Video File":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4"])

    if uploaded_video is not None:
        uploads_dir = os.path.join("uploads")
        video_path = os.path.join(uploads_dir, uploaded_video.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        st.video(video_path)

        if st.button("Start Detection"):
            st.warning("Processing detection...")

            cap = cv2.VideoCapture(video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_filename = os.path.splitext(uploaded_video.name)[0] + ".mp4"
            out_path = os.path.join("output", output_filename)

            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (frame_width, frame_height),
            )

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()

                if not success:
                    break

                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}")
                st.text(f"Processing frame {frame_count}/{total_frames}")
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

                with st.spinner("Waiting for 5 seconds!"):
                    time.sleep(5)

            downloadButton()

elif option == "Open Webcam":
    st.text("Press 'Start' to open the webcam")
    if st.button("Start"):
        cap = cv2.VideoCapture(0)
        cap.set(3, frame_width)
        cap.set(4, frame_height)

        while True:
            success, frame = cap.read()

            if not success:
                break

            frame = processFrame(frame)

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
