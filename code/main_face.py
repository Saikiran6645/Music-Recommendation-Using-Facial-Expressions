# Importing the required libraries
import cv2
from keras.models import load_model # type: ignore
import numpy as np
from youtubesearchpython import VideosSearch # Import YouTube search library
import webbrowser # Import webbrowser to open links
import os
import time # Import time for delays and tracking

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration ---
MODEL_PATH = "model/fer2013_mini_XCEPTION.102-0.66.hdf5"
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_CONFIRMATION_DURATION = 1 # Seconds to observe for stable emotion
WINDOW_NAME = 'Facial Expression Recognition - Press Q to Quit'

# --- Load Model ---
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please ensure the model file exists and Keras/TensorFlow is installed correctly.")
    exit()

# --- Check Haar Cascade ---
if not os.path.exists(HAAR_CASCADE_PATH):
    print(f"Error: Haar Cascade file not found at {HAAR_CASCADE_PATH}")
    print("Please ensure OpenCV is installed correctly and the cascades are available.")
    exit()
face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

# --- Emotion Mapping ---
emotion_youtube_query = {
    "Happy": "happy upbeat music reaction", # Added "reaction" for variety maybe
    "Sad": "sad emotional songs compilation",
    "Angry": "angry metal music mix",
    "Neutral": "calm ambient background music",
    "Surprise": "surprising epic music reveal",
    "Fear": "calming relaxing music for anxiety",
    "Disgust": "heavy industrial music"
    # Add or adjust queries as you like
}
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- State Variables for Emotion Stability ---
current_tracking_emotion = None
emotion_start_time = None
stable_emotion_confirmed = False # Flag to indicate we found and acted on an emotion

# --- Function to Search and Open YouTube Video ---
# Modified to return True if successful, False otherwise
def play_video(final_emotion):
    final_emotion = final_emotion.capitalize()
    if final_emotion in emotion_youtube_query:
        search_query = emotion_youtube_query[final_emotion]
        print(f"\nStable emotion confirmed: {final_emotion}.")
        print(f"Searching YouTube for: '{search_query}'")
        try:
            videosSearch = VideosSearch(search_query, limit=1)
            results = videosSearch.result()
            if results and results['result']:
                video_url = results['result'][0]['link']
                print(f"Opening video: {video_url}")
                webbrowser.open(video_url)
                print("Video opened. Exiting application...")
                return True # Indicate success
            else:
                print(f"No YouTube video found for query: '{search_query}'")
                return False # Indicate failure
        except Exception as e:
            print(f"Error searching or opening YouTube video: {e}")
            return False # Indicate failure
    else:
        print(f"No YouTube search query mapped for emotion: {final_emotion}")
        return False # Indicate failure


# --- Setup Video Capture ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture device (camera).")
    print("Please ensure a camera is connected and drivers are installed.")
    exit()

print(f"Starting facial expression recognition. Observing for {EMOTION_CONFIRMATION_DURATION} seconds for a stable emotion...")
print("Look at the camera. Press 'q' to quit early.")

# --- Main Loop ---
while True:
    # --- Check if we already confirmed and played ---
    if stable_emotion_confirmed:
        break # Exit the loop immediately

    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from camera.")
        time.sleep(0.5)
        continue

    frame = cv2.flip(frame, 1) # Mirror view
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    detected_emotion_in_frame = None # Emotion detected in *this specific frame*

    # --- Process Detected Faces ---
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        if roi_gray.size == 0: continue # Skip empty ROI

        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        try:
            prediction = model.predict(roi, verbose=0)
            emotion_index = np.argmax(prediction[0])
            detected_emotion_in_frame = emotions[emotion_index] # Get the primary emotion

            # --- Draw on Frame ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, detected_emotion_in_frame, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Only process the first detected face for stability check for simplicity
            break # Exit the inner face loop after processing the first face

        except Exception as e:
            print(f"Error during model prediction: {e}")
            continue # Skip this face

    # --- Emotion Stability Logic ---
    current_time = time.time()

    if detected_emotion_in_frame:
        if detected_emotion_in_frame == current_tracking_emotion:
            # Emotion is consistent, check duration
            if emotion_start_time and (current_time - emotion_start_time >= EMOTION_CONFIRMATION_DURATION):
                # CONFIRMED STABLE EMOTION!
                if play_video(current_tracking_emotion):
                    stable_emotion_confirmed = True # Set flag to exit loop
                else:
                    # If playing failed, reset tracking to potentially try again later or quit
                    current_tracking_emotion = None
                    emotion_start_time = None
                    print("Failed to play video. Resetting emotion tracking.")

        else:
            # New or changed emotion detected, start tracking it
            print(f"Tracking new emotion: {detected_emotion_in_frame}...")
            current_tracking_emotion = detected_emotion_in_frame
            emotion_start_time = current_time
    else:
        # No face/emotion detected in this frame, reset tracking
        if current_tracking_emotion is not None:
             print("Face lost or emotion unclear. Resetting tracking.")
        current_tracking_emotion = None
        emotion_start_time = None

    # --- Display Tracking Status ---
    status_text = "Status: Waiting..."
    if current_tracking_emotion and emotion_start_time:
        elapsed = current_time - emotion_start_time
        status_text = f"Status: Tracking '{current_tracking_emotion}' for {elapsed:.1f}s / {EMOTION_CONFIRMATION_DURATION}s"
    cv2.putText(frame, status_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # --- Display Frame ---
    cv2.imshow(WINDOW_NAME, frame)

    # --- Check for Quit Key ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting manually...")
        break

# --- Release Resources ---
cap.release()
cv2.destroyAllWindows()
print("Application finished.")
