## Music-Recommendation-Using-Facial-Expressions
### Project Overview

This project is a Python-based application that uses OpenCV for real-time facial detection and a pre-trained deep learning model (fer2013_mini_XCEPTION.102-0.66.hdf5) to recognize and analyze facial expressions. By capturing live video feed from the user’s webcam, it identifies the user’s emotions—such as happiness, sadness, anger, or neutrality—based on facial cues.

Once the emotion is detected, the application constructs a YouTube search query tailored to the identified mood. Using the webbrowser module, the application automatically opens relevant YouTube search results in the user’s default browser, allowing them to access music that aligns with their current emotional state. The requests library further supports this functionality by enabling API interactions for a smoother YouTube search experience.

This project combines elements of computer vision and deep learning with web integration to create a personalized and interactive music recommendation system. It demonstrates the potential of AI-powered emotion detection in real-world applications, where user experience can be enhanced through real-time responsiveness and intelligent content recommendations.

## 🔧 Installation

Follow these steps to set up the project locally:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Saikiran6645/Music-Recommendation-Using-Facial-Expressions.git
   cd Music-Recommendation-Using-Facial-Expressions
   pip install requirements.txt

### How to Run

1.  **Execute the Script:**
    ```bash
    python main_face.py
    ```

### Tech Stack & Libraries

- Python: As the primary programming language for its versatility and extensive libraries.
- OpenCV: For real-time image and video processing, including facial detection.
- TensorFlow and Keras: For building and training the deep learning model to recognize facial expressions.
- fer2013_mini_XCEPTION.102-0.66.hdf5: A pre-trained model for facial emotion recognition.
- requests: For making HTTP requests to interact with web APIs (e.g., YouTube search).

