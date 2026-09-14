import cv2
import torch
import numpy as np
import os
import requests
from ultralytics import YOLO
import supervision as sv  # Import supervision library for ByteTrack
from concurrent.futures import ThreadPoolExecutor
import time    # Format the time into a readable string (e.g., "000012.34" for 12.34 seconds)

# Telegram credentials
TOKEN = '7816125652:AAHUVUbi7RQfqAErpd4Xs5syL_SjXdBXywY'
ID_CONTACT = 'CHANGE-UR-ID'
TELEGRAM_URL = f'https://api.telegram.org/bot{TOKEN}/sendPhoto'

# Load YOLOv8 model
model = YOLO("yolov8s_trained.pt")  # Path to your trained model

# Class Names (update based on your training labels)
CLASS_NAMES = ["helm", "motor", "nohelmet"]

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
    cap = cv2.VideoCapture(0)  # Using default webcam
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
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

# Fungsi untuk menggabungkan gambar dengan padding hitam
def combine_with_black_padding(img1, img2):
    """Combine two images side by side with black padding to match heights."""
    height1, width1 = img1.shape[:2]
    height2, width2 = img2.shape[:2]

    # Tentukan tinggi maksimum
    max_height = max(height1, height2)

    # Buat kanvas hitam untuk padding
    canvas1 = np.zeros((max_height, width1, 3), dtype=np.uint8)
    canvas2 = np.zeros((max_height, width2, 3), dtype=np.uint8)

    # Tempelkan gambar asli di tengah kanvas hitam
    canvas1[:height1, :width1] = img1
    canvas2[:height2, :width2] = img2

    # Gabungkan secara horizontal
    combined_img = np.hstack((canvas1, canvas2))
    return combined_img

# Loop deteksi
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video.")
        break
    
    video_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / (video_info.fps if video_input == 'file' else 30)
    formatted_time = f"{video_time:012.2f}"

    # YOLOv8 Inference
    results = model(frame, verbose=False, conf=0.5, iou=0.7) #atur conf sesuai yang diinginkan, semakin tinggi conf maka deteksi hanya mengambil yang lebih pasti, resikonya banyak yang tidak terdeteksi, rendah conf akan semakin mempengaruhi banyak hal yang seharusnya tidak masuk malah terdeteksi. Atur IoU, semakin besar iout, maka deteksi untuk overlap akan minim dikarenakan yang akan diambil yang memiliki true positif yang diatas batas iou
    result = results[0]
    detections = result.boxes

    motor_tracker = {}
    no_helmet_tracker = {}

    if detections is not None and len(detections) > 0:
        detections_sv = sv.Detections.from_ultralytics(result)
        tracked_detections = byte_track.update_with_detections(detections_sv)

        for xyxy, class_id, conf, tracker_id in zip(
            tracked_detections.xyxy, tracked_detections.class_id, tracked_detections.confidence, tracked_detections.tracker_id
        ):
            class_id = int(class_id)
            x1, y1, x2, y2 = map(int, xyxy)

            # Draw bounding boxes and labels on the video
            label = f"{CLASS_NAMES[class_id]} ID:{tracker_id} Conf:{conf:.2f} (x:{x1}, y:{y1})" 
            # Draw bounding box and labels
            if class_id == 0:  # Helm
                color = (0, 255, 0)  # Green
            elif class_id == 1:  # Motor
                color = (255, 255, 0)  # Cyan
            elif class_id == 2:  # No Helmet
                color = (0, 0, 255)  # Red
            elif class_id == 3:  # Plat
                color = (255, 255, 255)  # White            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.2, color, 1)


            # Simpan koordinat objek motor dan tanpa helm
            if class_id == 1:  # Motor
                motor_tracker[tracker_id] = (x1, y1, x2, y2)
            elif class_id == 2:  # No Helmet
                no_helmet_tracker[tracker_id] = (x1, y1, x2, y2)

        # Proses "No Helmet" dan cari pasangan "Motor"
        for no_helmet_id, no_helmet_coords in no_helmet_tracker.items():
            nx1, ny1, nx2, ny2 = no_helmet_coords
            motor_found = False  # Flag jika motor ditemukan

            for motor_id, motor_coords in motor_tracker.items():
                mx1, my1, mx2, my2 = motor_coords

                # Periksa apakah motor berada di bawah "No Helmet"
                if mx1 <= nx1 <= mx2 and my1 >= ny2:
                    motor_found = True

                    # **Perpanjang bounding box motor ke atas hingga y1 No Helmet**
                    adjusted_my1 = ny1  # Sesuaikan y motor ke atas sejajar y No Helmet

                    # Crop gambar (dari atas helm hingga bawah motor)
                    combined_y1 = max(0, adjusted_my1)  # Pastikan tidak negatif
                    motor_img = frame[combined_y1:my2, mx1:mx2]

                    # Screenshot area "No Helmet"
                    no_helmet_img = frame[ny1:ny2, nx1:nx2]

                    # Gabungkan dengan padding hitam (jika perlu)
                    combined_img = combine_with_black_padding(no_helmet_img, motor_img)

                    # Simpan gambar gabungan
                    file_name = f"{counter}_nohelmet_motor_{formatted_time}.jpg"
                    cv2.imwrite(file_name, combined_img)
                    caption_text = f"{counter}. Motor tanpa helm ID:{motor_id}_{formatted_time}, Conf:{conf:.2f}, (x:{x1,x2}, y:{y1,y2})"

                    print(f"✅ Image saved: {caption_text}")

                    # Kirim ke Telegram
                    executor.submit(send_to_telegram, file_name, caption_text)
                    counter += 1
                    break

            if not motor_found:
                print(f"❌ No motor found for 'No Helmet' ID: {no_helmet_id}")

    # Tampilkan frame
    resized_frame = cv2.resize(frame, (800, 600))
    cv2.imshow("Helmet Detection", resized_frame)

    # Keluar dengan menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
executor.shutdown()
