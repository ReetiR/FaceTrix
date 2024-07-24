import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch
from sklearn.neighbors import KNeighborsClassifier

# Function to speak text using Windows SAPI
def speak(text):
    speak_engine = Dispatch("SAPI.SpVoice")
    speak_engine.Speak(text)

# Open webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load the names and faces data
names_file = 'names.pkl'
faces_file = 'faces.pkl'

# Verify that the files exist
if not os.path.exists(names_file):
    print(f"Error: The file {names_file} does not exist.")
    exit(1)

if not os.path.exists(faces_file):
    print(f"Error: The file {faces_file} does not exist.")
    exit(1)

with open(names_file, 'rb') as f:
    LABELS = pickle.load(f)

with open(faces_file, 'rb') as f:
    FACES = pickle.load(f)

# Initialize and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=min(5, len(FACES)))
knn.fit(FACES, LABELS)

# Load background image
image_bgm_path = 'C:/Users/dell/Desktop/c++ programming/dataa/dataa/reeti.png'
image_bgm = cv2.imread(image_bgm_path)

COL_NAMES = ['NAME', 'TIME']

if image_bgm is None:
    print(f"Error: Background image not found at {image_bgm_path}.")
    exit(1)

if not cap.isOpened():
    print("Error: Could not open video source.")
else:
    while True:
        ret, frame = cap.read()

        # Check if frame was read correctly
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        # List to store names detected in the current frame
        current_frame_names = set()
        
        # Draw rectangles around faces and make predictions
        for (x, y, w, h) in faces:
            # Draw a customized rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle with thickness 2
            
            crop_img = frame[y:y + h, x:x + w]
            resize_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resize_img)
            
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d_%m_%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H_%M_%S")
            
            # Draw text with background
            label = str(output[0])
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Red text with thickness 2
            
            # Add name to the list for this frame
            current_frame_names.add(label)
        
        # Resize the frame to fit the region in the background image
        target_height = 415  # Height of the region in the background image
        target_width = 640   # Width of the region in the background image
        
        if frame.shape[0] != target_height or frame.shape[1] != target_width:
            frame_resized = cv2.resize(frame, (target_width, target_height))
        else:
            frame_resized = frame
        
        # Overlay the frame onto the background image
        y1, y2 = 162, 162 + target_height
        x1, x2 = 55, 55 + target_width
        image_bgm[y1:y2, x1:x2] = frame_resized
        
        # Display the background image with the overlaid frame
        if image_bgm is None or image_bgm.size == 0:
            print("Error: Background image is empty.")
            break
        
        if image_bgm.shape[0] > 0 and image_bgm.shape[1] > 0:
            cv2.imshow("Frame", image_bgm)
        else:
            print("Error: Frame dimensions are invalid.")
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on 'q' key press
        if key == ord('q'):
            break
        
        # Take attendance on 'o' key press
        if key == ord('o'):
            speak("Attendance Taken.")
            time.sleep(5)
            
            attendance = [(name, timestamp) for name in current_frame_names]
            filename = f"Attendance/Attendance_{date}.csv"
            
            # Check if file exists
            file_exists = os.path.isfile(filename)
            
            # Write attendance to CSV file
            with open(filename, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write headers if file does not exist
                if not file_exists:
                    writer.writerow(COL_NAMES)
                
                for record in attendance:
                    writer.writerow(record)
                
    cap.release()
    cv2.destroyAllWindows()
