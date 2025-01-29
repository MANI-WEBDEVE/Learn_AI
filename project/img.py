import cv2
import os

# Create directory if it does not exist
directory = 'images'
if not os.path.exists(directory):
    os.makedirs(directory)

# Capture four images automatically
for i in range(4):
    # Access the camera
    camera = cv2.VideoCapture(0)
   

    # Check if camera opened successfully
    if not camera.isOpened():
        print("Error: Could not open camera.")
        exit()

    # Capture frame-by-frame
    ret, frame = camera.read()

    # Save the captured frame to the specified directory
    cv2.imwrite(os.path.join(directory, f'captured_image_{i+1}.jpg'), frame)
    print(f"Image {i+1} saved in:", directory)

    # Release the camera
    camera.release()

# Destroy all windows
cv2.destroyAllWindows()
