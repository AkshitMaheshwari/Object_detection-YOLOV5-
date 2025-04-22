import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import random
import time

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

model = load_model()

@st.cache_resource
def generate_colors():
    return {label: [random.randint(0, 255) for _ in range(3)] for label in model.names}

colors = generate_colors()

def detect_objects(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)
    detections = results.pandas().xyxy[0]

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf, label = row['confidence'], row['name']
        if label not in colors:
            colors[label] = [random.randint(0, 255) for _ in range(3)]
        color = colors[label]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label_text = f"{label} ({conf:.2f})"
        font_scale = 0.70
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        cv2.rectangle(frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)

        cv2.putText(frame, label_text, (x1, y1 - 2), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    return frame
st.title("Real-Time Object and Video Detection with YOLOv5")
st.write("""
This app uses the YOLOv5 model for real-time object detection in both video streams and images.
""")

st.sidebar.title("Options")
mode = st.sidebar.radio("Select Mode:", ["Real-Time Camera", "Upload Image or Video"])

if mode == "Real-Time Camera":
    st.write("### Real-Time Object Detection")
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False

    start = st.sidebar.button("Start Camera")
    stop = st.sidebar.button("Stop Camera")

    if start:
        st.session_state.camera_active = True
    if stop:
        st.session_state.camera_active = False

    frame_placeholder = st.empty()

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        st.write("Camera is active. Press 'Stop Camera' to end.")

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera.")
                break

            detected_frame = detect_objects(frame)
            frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            time.sleep(0.05)  

            if not st.session_state.camera_active:
                break

        cap.release()
        st.write("Camera stopped.")

elif mode == "Upload Image or Video":
    st.write("### Object Detection on Uploaded Image or Video")
    uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "WEBP", "mp4", "avi", "mov"])

    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type.startswith("image/"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            image_cv = np.array(image)
            if image_cv.shape[2] == 4:
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGBA2RGB)

            detected_image = detect_objects(image_cv)
            st.image(detected_image, caption="Detection Results", use_column_width=True)

        elif file_type.startswith("video/"):
            tfile = f"temp_video.mp4"
            with open(tfile, "wb") as f:
                f.write(uploaded_file.read())

            cap = cv2.VideoCapture(tfile)
            stframe = st.empty()
            st.write("### Processing Video...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                detected_frame = detect_objects(frame)
                frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)
                time.sleep(0.03)

            cap.release()
            st.write("### Video Processing Complete.")
