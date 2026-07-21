import cv2
import time
import numpy as np

cap = cv2.VideoCapture(0)

frame_buffer = []

fps = 15
interval = 1 / fps

last_capture_time = time.time()

print("Press Q to quit")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Cannot access webcam")
        break

    current_time = time.time()

    if current_time - last_capture_time >= interval:

        frame_buffer.append(frame.copy())

        print(f"Frames Stored: {len(frame_buffer)}")

        last_capture_time = current_time

    cv2.imshow("15 FPS Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\nCapture Finished")
print("Total Frames Stored:", len(frame_buffer))

if len(frame_buffer) > 0:
    print("Shape of first frame:", frame_buffer[0].shape)