import cv2
import os

# Create directory if it does not exist
directory = 'videos'
if not os.path.exists(directory):
    os.makedirs(directory)

# Access the laptop's built-in camera
camera = cv2.VideoCapture(0)  # Use 0 to access the default camera (usually built-in webcam)

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(directory, 'captured_video.avi'), fourcc, 20.0, (640, 480))

# Record video for a certain duration (in seconds)
duration = 10  # Adjust the duration as needed

start_time = cv2.getTickCount()
while cv2.getTickCount() - start_time < duration * cv2.getTickFrequency():
    ret, frame = camera.read()
    if not ret:
        break
    out.write(frame)
    cv2.imshow('Recording...', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
camera.release()
out.release()
cv2.destroyAllWindows()

print("Video saved in:", directory)
