import streamlit as st
import cv2
import numpy as np
import datetime

# Function to perform animal detection and save frames
def perform_animal_detection(frame, net, classes, ani):
    height, width, _ = frame.shape

    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Process YOLO output
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # Adjust this threshold as needed
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-maximum suppression to eliminate redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for index in indices:
        index = index  # No need to access [0] here
        box = boxes[index]
        x, y, w, h = box
        label = str(classes[class_ids[index]])
        confidence = confidences[index]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        matching_names = []
        for name in ani:
            if label.lower() in name.lower():
                matching_names.append(name)
                save_detected_frame(label, frame)

    return frame

# Function to save the detected frame
def save_detected_frame(label, frame):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detected_animal[label].append(current_time)

    # Overlay timestamp on the image
    timestamp_text = f"{label} - {current_time}"
    cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save entire detected frame as image
    frame_path = f"{detected_animal}/{label}_{current_time.replace(':', '-')}.jpg"
    cv2.imwrite(frame_path, frame)

# Streamlit UI
st.title("Animal Detection")

# Replace with the path to your YOLO weights and configuration files
net = cv2.dnn.readNet("D:/VSCode/animal/yolov3-tiny.weights", "D:/VSCode/animal/yolov3-tiny.cfg")

ani = ["cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]

# Load classes
classes = []
with open("D:/VSCode/animal/coco.names", "r") as f:
    classes = f.read().splitlines()

# Replace with the path to your video
video_path = "D:/VSCode/animal/horse_-_26296 (720p).mp4"
cap = cv2.VideoCapture(video_path)

# Initialize detected_animal dictionary
detected_animal = {name: [] for name in ani}

# Create a placeholder for the video player
video_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform animal detection and update the video player
    processed_frame = perform_animal_detection(frame, net, classes, ani)
    video_placeholder.image(processed_frame, channels="BGR")

# Close the video capture
cap.release()

st.write("Video processing complete!")
