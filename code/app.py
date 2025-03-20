
import streamlit as st
import cv2
from keras.models import load_model  # type: ignore # Assuming you have Keras installed
import numpy as np
import webbrowser
import requests
import re
import os
import time


#final_emotion = None
# Function to play the first song from youtube queries
def play_first_song(final_emotion):
    try:
        search_query = f"https://www.youtube.com/results?search_query={final_emotion}+background+tunes"
        
        # to fetch the search results page
        response = requests.get(search_query)
        
        # HTTP status code 200 = request was successful 
        if response.status_code != 200:
            print("Failed to retrieve YouTube search results. Status code:", response.status_code)
            return
        
        html_content = response.text
        
        match = re.search(r'/watch\?v=([^\"]+)', html_content)
        if match:
            video_id = match.group(1)
            #video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_url = f"https://www.youtube.com/watch?v={video_id.encode('utf-8').decode('unicode_escape')}"
            
            # printing the video URL for debugging purposes
            print("Opening YouTube video:", video_url)
            
            # opening the video in the default web browser
            webbrowser.open(video_url)
        else:
            print("No video found for the given query.")

    except requests.RequestException as e:
        print("An error occurred while connecting to YouTube:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)

# Define the list of emotions (modify if your model uses different classes)
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model("model/fer2013_mini_XCEPTION.102-0.66.hdf5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_emotion():
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    emotion_label = None
    placeholder = st.empty()
    #run1 = st.checkbox("Play Song", key ="2")

    while True:
        ret, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract the region of interest (ROI) which is the face
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

            # Normalize the pixel values
            roi = roi_gray / 255.0

            # Reshape the image for the model
            roi = np.reshape(roi, (1, 64, 64, 1))

            # Make a prediction using the pre-trained model
            prediction = model.predict(roi)
            emotion_label = emotions[np.argmax(prediction)]
            # Draw a rectangle around the face and display the predicted emotion
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            button_text = "Capture Emotion"
            cv2.putText(frame, button_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        FRAME_WINDOW.image(frame)


def predict_emotion_button():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))
        prediction = model.predict(roi)
        emotion_label = emotions[np.argmax(prediction)]
        play_first_song(emotion_label)
        st.write("Song Detected is:", emotion_label)
        print("Song Detected is:", emotion_label)
        

def main():
  """
  The main function of the Streamlit app.
  """

  st.title("Facial Emotion Recognition App")
  st.write("This app detects your facial expression and displays the predicted emotion.")
  if st.button("Play Song"):
    predict_emotion_button()
  run = st.checkbox("Start Webcam", key ="1")
  camera = cv2.VideoCapture(0)
  #frame_width = st.slider("Camera Width", min_value=300, max_value=1000)
  FRAME_WINDOW = st.image([])

  while run:
    predict_emotion()
    time.sleep(20)
  else:
    camera.release()

if __name__ == '__main__':
  main()
