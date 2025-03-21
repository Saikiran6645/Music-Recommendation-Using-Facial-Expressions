import cv2
import PySimpleGUI as sg
import numpy as np
import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from keras.models import load_model # type: ignore
from keras.optimizers import Adam # type: ignore
import os
from threading import Thread

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model without compilation
model = load_model('code/model/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

# Recompile with a new optimizer
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

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

def detect_emotion(frame, face_cascade):
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
        return frame, emotion_label
    return None, None

def video_thread(window):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    current_emotion = None
    last_emotion_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (640, 480))
            current_time = time.time()
            
            if current_time - last_emotion_time >= 10:
                frame_with_faces, current_detected_emotion = detect_emotion(frame, face_cascade)
                if frame_with_faces is not None:
                    imgbytes = cv2.imencode('.png', frame_with_faces)[1].tobytes()
                    window['-IMAGE-'].update(data=imgbytes)
                    
                    if current_detected_emotion != current_emotion:
                        window['-EMOTION-'].update(value=f'Detected Emotion: {current_detected_emotion}')
                        current_emotion = current_detected_emotion
                        last_emotion_time = current_time
        
        cv2.waitKey(1)

def play_song_with_emotion(emotion):
    emotion = emotion.capitalize()
    if emotion in emotion_playlist:
        uri = emotion_playlist[emotion]
        devices = sp.devices()["devices"]
        if devices:
            device_id = devices[0]["id"]
            sp.start_playback(device_id=device_id, context_uri=uri)
        else:
            print("No active Spotify device found.")
    else:
        print("No playlist mapped for this emotion.")

def gui_thread():
    layout = [
        [sg.Image(filename='', key='-IMAGE-')],
        [sg.Text('Detected Emotion: ', key='-EMOTION-')],
        [sg.Button('Capture Emotion', button_color=('black', 'orange'), key='-CAPTURE-EMOTION-')],
        [sg.Button('Play Song', button_color=('black', 'green'), key='-PLAY-')],
        [sg.Text(size=(30, 1), key='-RETURN-VALUE-')]
    ]

    window = sg.Window('Facial Expression Recognition', layout)
    Thread(target=video_thread, args=(window,), daemon=True).start()
    current_emotion = None

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        elif event == '-CAPTURE-EMOTION-':
            current_emotion = window['-EMOTION-'].DisplayText.split(":")[1].strip()
            window['-RETURN-VALUE-'].update(value=f'Detected Emotion: {current_emotion}')
        elif event == '-PLAY-' and current_emotion:
            play_song_with_emotion(current_emotion)

    window.close()

if __name__ == "__main__":
    gui_thread()
