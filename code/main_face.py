# Importing the required libraries
import cv2
from keras.models import load_model # type: ignore
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the emotion detection model
try:
    model = load_model("code/model/fer2013_mini_XCEPTION.102-0.66.hdf5")
except ValueError as e:
    print("Error loading model:", e)
    exit()

# Spotify Authentication
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    redirect_uri="http://localhost:8888/callback",
    scope="user-modify-playback-state,user-read-playback-state"
))

# Mapping emotions to Spotify playlist URIs
emotion_playlist = {
    "Happy": "spotify:playlist:37i9dQZF1DXdPec7aLTmlC",
    "Sad": "spotify:playlist:37i9dQZF1DX7qK8ma5wgG1",
    "Angry": "spotify:playlist:37i9dQZF1DX3YSRoSdA634",
    "Neutral": "spotify:playlist:37i9dQZF1DX3Ogo9pFvBkY"
}

# Function to play songs on Spotify based on detected emotion
def play_music(final_emotion):
    final_emotion = final_emotion.capitalize()
    if final_emotion in emotion_playlist:
        uri = emotion_playlist[final_emotion]
        try:
            devices = sp.devices()
            if devices['devices']:
                device_id = devices['devices'][0]['id']
                sp.start_playback(device_id=device_id, context_uri=uri)
            else:
                print("No active Spotify devices found.")
        except Exception as e:
            print("Error playing music:", e)
    else:
        print("No playlist mapped for this emotion.")

# Facial expression recognition setup
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))

        prediction = model.predict(roi)
        emotion_label = emotions[np.argmax(prediction)]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        play_music(emotion_label)

    cv2.imshow('Facial Expression Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
