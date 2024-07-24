import cv2
import pickle
import numpy as np
import os

# Ensure the directory exists
data_path = 'dataa/'
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f"Created directory: {data_path}")

# Open webcam
cap = cv2.VideoCapture(0)

# Load Haar Cascade
facedetect = cv2.CascadeClassifier(r'C:\Users\dell\Desktop\c++ programming\dataa\haarcascade_frontalface_default.xml')

faces_data = []
i = 0
name = input("Enter Your Name: ")

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
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 225), 1)
            crop_img = frame[y:y + h, x:x + w]
            resize_img = cv2.resize(crop_img, (50, 50))
            
            if len(faces_data) < 100 and i % 10 == 0:
                faces_data.append(resize_img)
        
        i += 1
        
        # Display the number of faces collected
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 225), 1)

        # Display the frame
        if frame is None or frame.size == 0:
            print("Error: Frame is empty.")
            break
        
        if frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow("frame", frame)
        else:
            print("Error: Frame dimensions are invalid.")
        
        # Exit on 'q' key press or if 100 faces are collected
        if cv2.waitKey(1) & 0xFF == ord('q') or len(faces_data) == 100:
            break

    cap.release()
    cv2.destroyAllWindows()

# Convert faces_data to a numpy array and reshape
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)
print(f"Faces Data Shape: {faces_data.shape}")

# Paths for saving files
names_file = os.path.join(data_path, 'names.pkl')
faces_file = os.path.join(data_path, 'faces_data.pkl')

# Handle names file
if not os.path.isfile(names_file):
    names = [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
    print(f"Created and saved {names_file}")
else:
    with open(names_file, 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
    with open(names_file, 'wb') as f:
        pickle.dump(names, f)
    print(f"Updated and saved {names_file}")

# Handle faces data file
if not os.path.isfile(faces_file):
    with open(faces_file, 'wb') as f:
        pickle.dump(faces_data, f)
    print(f"Created and saved {faces_file}")
else:
    with open(faces_file, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open(faces_file, 'wb') as f:
        pickle.dump(faces, f)
    print(f"Updated and saved {faces_file}")
