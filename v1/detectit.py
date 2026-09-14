## SET UR ID TELEGRAM BEFORE USE THIS ##
import cv2
import torch
import numpy as np
import os
import requests
from ultralytics import YOLO
import supervision as sv  # Import supervision library for ByteTrack
from concurrent.futures import ThreadPoolExecutor
import time    # Format the time into a readable string (e.g., "000012.34" for 12.34 seconds)

# Telegram Bot Configuration
ID_CONTACT = 'CHANGE_UR_ID' #Get ur ID from @username_to_id_bot -> /start -> Click 'user' -> choose it -> get the 'id' 
TOKEN = '7816125652:AAHUVUbi7RQfqAErpd4Xs5syL_SjXdBXyw' # http://t.me/vbdh_bot open with that 'user' -> 'user' click /start -> run this program 
TELEGRAM_URL = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'

# Load YOLOv8 model
model = YOLO("yolov8s_retrained.pt")  # Path to your trained model

# Class Names (update based on your training labels)
CLASS_NAMES = ["helm", "motor", "nohelmet", "plat"]

# Ask user for video input type
video_input = input("Enter 'file' for video file or 'webcam' for webcam: ").strip().lower()

if video_input == 'file':
    video_path = input("Enter the path to your video file: ").strip()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        exit()
    # ByteTrack initialization for video file
    video_info = sv.VideoInfo.from_video_path(video_path=video_path)
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)
elif video_input == 'webcam':
    # Ask user for webcam input option (default 0 or other options like 1, 2, 3)
    camera_id = input("Enter webcam ID (0 for default, 1 for external webcam 1, 2 for external webcam 2, etc.): ").strip()

    # Set default to 0 if no valid input is provided
    if camera_id == '' or not camera_id.isdigit():
        camera_id = 0
    else:
        camera_id = int(camera_id)

    # Open the chosen webcam
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Cannot open webcam {camera_id}.")
        exit()

    # ByteTrack initialization for webcam (default FPS assumption)
    byte_track = sv.ByteTrack(frame_rate=30)  # Default frame rate for webcam (can be adjusted)

else:
    print("Invalid input. Exiting.")
    exit()

# Allow manual resize of window
cv2.namedWindow("Helmet Detection", cv2.WINDOW_NORMAL)

# Tracker status
motor_tracker = {}
no_helmet_tracker = {}
processed_logs = set()
counter = 1

# Thread Pool for Parallel Uploading
executor = ThreadPoolExecutor(max_workers=5)

def send_to_telegram(image_path, caption):
    """Function to send image to Telegram."""
    with open(image_path, 'rb') as file:
        files = {'photo': file}
        data = {'chat_id': ID_CONTACT, 'caption': caption}
        response = requests.post(TELEGRAM_URL, files=files, data=data)
    if response.status_code == 200:
        print(f"✅ Sent to Telegram: {caption}")
    else:
        print(f"❌ Failed to send: {response.json()}")
    os.remove(image_path)

# Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    
    video_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / (video_info.fps if video_input == 'file' else 30)  # Current time in seconds
    formatted_time = f"{video_time:012.2f}"

    # YOLOv8 Inference
    results = model(frame, verbose=False)
    result = results[0]
    detections = result.boxes

    if detections is not None and len(detections) > 0:
        detections_sv = sv.Detections.from_ultralytics(result)
        tracked_detections = byte_track.update_with_detections(detections_sv)

        for xyxy, class_id, tracker_id in zip(
            tracked_detections.xyxy, tracked_detections.class_id, tracked_detections.tracker_id
        ):
            class_id = int(class_id)
            x1, y1, x2, y2 = map(int, xyxy)

            # Draw bounding boxes and labels on the video
            label = f"{CLASS_NAMES[class_id]} ID:{tracker_id} (x:{x1}, y:{y1})"
            # Define colors for each class
            colors = {
                0: (0, 255, 0),   # Green for helm
                1: (255, 0, 0),   # Blue for no_helmet
                2: (0, 0, 255),   # Red for motor
                3: (255, 255, 255)  # White for plat
            }

            # Use the color based on class_id
            color = colors.get(class_id, (255, 255, 255))  # Default to white if class_id is not found  
            
                      
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Track motor and no-helmet
            if class_id == 1:  # Motor
                motor_tracker[tracker_id] = (x1, y1, x2, y2)
            elif class_id == 2:  # No Helmet
                no_helmet_tracker[tracker_id] = (x1, y1, x2, y2)
            elif class_id == 3:  # Plat
                for motor_id, motor_coords in motor_tracker.items():
                    mx1, my1, mx2, my2 = motor_coords
                    if mx1 <= x1 <= mx2 and my1 <= y1 <= my2:
                        for nh_id, nh_coords in no_helmet_tracker.items():
                            if mx1 <= nh_coords[0] <= mx2 and my1 <= nh_coords[1] <= my2:
                                # Crop motor and plate images
                                motor_img = frame[my1:my2, mx1:mx2]
                                plate_img = frame[y1:y2, x1:x2]

                                # Combine images on a canvas
                                motor_h, motor_w = motor_img.shape[:2]
                                plate_h, plate_w = plate_img.shape[:2]
                                canvas_height = max(motor_h, plate_h)
                                canvas_width = motor_w + plate_w
                                combined_img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
                                combined_img[:motor_h, :motor_w] = motor_img
                                combined_img[:plate_h, motor_w:motor_w + plate_w] = plate_img

                                # Generate naming
                                motor_number = f"b{motor_id:03d}"
                                plate_number = f"d{tracker_id:03d}"  # Or you can extract actual plate text if OCR is implemented
                                caption_text = f"{counter}.Motor {motor_number} tanpa helm dengan plat {plate_number}"
                                file_name = f"{counter}.motor_{motor_number}_tanpa_helm_plat_{plate_number}_{formatted_time}.jpg"

                                # Add caption to the image
                                cv2.putText(combined_img, caption_text, (20, 50),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                                # Save temporarily
                                temp_filename = f"temp_{counter}.jpg"
                                cv2.imwrite(file_name, combined_img)
                                print(f"✅ Image saved: {caption_text}")

                                # Send image to Telegram in parallel
                                executor.submit(send_to_telegram, file_name, caption_text)

                                counter += 1

    # Display the video frame
    resized_frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Helmet Detection", resized_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
executor.shutdown()
