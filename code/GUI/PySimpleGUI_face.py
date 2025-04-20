import cv2
import numpy as np
import time
from keras.models import load_model  # type: ignore
from keras.optimizers import Adam  # type: ignore
import urllib.parse
import webbrowser
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model and compile
model = load_model('code/model/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotion(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi = roi_gray / 255.0
        roi = np.reshape(roi, (1, 64, 64, 1))
        prediction = model.predict(roi, verbose=0)
        emotion_label = emotions[np.argmax(prediction)]
        return emotion_label
    return None

def play_song_with_emotion(emotion):
    query = f"{emotion} mood songs"
    youtube_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}"
    print(f"\n🎶 Opening YouTube for: {query}")
    webbrowser.open(youtube_url)

def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    print("🟢 Starting Emotion Detection... (Press 'q' to quit)")

    last_detected_emotion = None
    last_detection_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("Live Feed", frame)

        current_time = time.time()
        if current_time - last_detection_time >= 10:  # Detect every 10 seconds
            emotion = detect_emotion(frame, face_cascade)
            if emotion and emotion != last_detected_emotion:
                print(f"\n🙂 Detected Emotion: {emotion}")
                play_song_with_emotion(emotion)
                last_detected_emotion = emotion
            last_detection_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
