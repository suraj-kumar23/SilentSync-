
import json
from collections import defaultdict
from difflib import get_close_matches
from flask import Flask, render_template, Response, jsonify, request
import cv2
import classifier  # Your sign language classifier module
import pyttsx3
import time
import threading
from googletrans import Translator
from deepface import DeepFace
from mtcnn import MTCNN
from collections import deque

app = Flask(__name__)

class WordSuggestor:
    def __init__(self):
        try:
            with open('word_freq.json') as f:
                self.word_freq = json.load(f)
        except FileNotFoundError:
            self.word_freq = {
                "hello": 100, "hi": 99, "goodbye": 98, "bye": 97, "yes": 96,
                "no": 95, "please": 94, "thank": 93, "you": 92, "me": 91, 
                "i": 90, "we": 89, "they": 88, "he": 87, "she": 86,
                "it": 85, "is": 84, "are": 83, "was": 82, "were": 81,
                "be": 80, "have": 79, "has": 78, "had": 77, "do": 76,
                "does": 75, "did": 74, "a": 73, "an": 72, "the": 71,
                "and": 70, "but": 69, "or": 68, "because": 67, "if": 66,
                "then": 65, "so": 64, "what": 63, "when": 62, "where": 61,
                "why": 60, "how": 59, "which": 58, "who": 57, "that": 56,
                "this": 55, "here": 54, "there": 53, "some": 52, "any": 51,
                "all": 50, "each": 49, "every": 48, "many": 47, "few": 46,
                "most": 45, "more": 44, "less": 43, "little": 42, "big": 41,
                "small": 40, "good": 39, "bad": 38, "like": 37, "want": 36,
                "need": 35, "know": 34, "think": 33, "say": 32, "tell": 31,
                "see": 30, "hear": 29, "go": 28, "come": 27, "get": 26,
                "make": 25, "take": 24, "give": 23, "use": 22, "find": 21,
                "work": 20, "play": 19, "eat": 18, "drink": 17, "sleep": 16,
                "home": 15, "day": 14, "time": 13, "people": 12, "thing": 11,
                "money": 10, "friend": 9, "family": 8, "name": 7, "food": 6,
                "water": 5, "house": 4, "car": 3, "book": 2, "phone": 1,
                "today": 55, "tomorrow": 50, "yesterday": 45, "now": 60, "later": 40,
                "always": 35, "never": 30, "often": 25, "sometimes": 20, "maybe": 15,
                "really": 48, "very": 52, "quite": 38, "just": 58, "still": 42,
                "again": 33, "already": 28, "almost": 23, "enough": 18, "too": 37,
                "here": 54, "there": 53, "around": 29, "away": 24, "back": 36,
                "down": 31, "forward": 19, "inside": 14, "outside": 17, "up": 39,
                "about": 46, "above": 27, "across": 22, "after": 32, "against": 16,
                "along": 13, "among": 11, "before": 34, "behind": 26, "below": 21,
                "beside": 10, "between": 12, "by": 41, "during": 15, "for": 51,
                "from": 49, "in": 65, "into": 20, "of": 62, "on": 59,
                "over": 35, "through": 28, "to": 63, "under": 25, "with": 56
            }
        self.user_history = defaultdict(int)
        
    def update_history(self, word):
        self.user_history[word.lower()] += 1
        
    def get_suggestions(self, prefix, num=5):
        candidates = [w for w in self.word_freq if w.lower().startswith(prefix.lower())]
        candidates.sort(key=lambda w: (
            self.user_history.get(w.lower(), 0),
            self.word_freq.get(w, 0)
        ), reverse=True)
        return candidates[:num]
    
    def get_corrections(self, word, num=3):
        if word.lower() in [w.lower() for w in self.word_freq]:
            return []
        return get_close_matches(word.lower(), self.word_freq.keys(), n=num)

class EmotionDetector:
    def __init__(self, min_confidence=70):
        # Configuration
        self.MIN_CONFIDENCE = min_confidence  # Minimum confidence for emotions
        self.SMOOTHING_WINDOW = 5  # Larger window for better accuracy
        self.DETECTOR = MTCNN()  # Use MTCNN for accurate face detection

        # Emotion Data
        self.EMOTION_COLORS = {
            'angry': (0, 0, 255),
            'disgust': (0, 102, 0),
            'fear': (255, 255, 0),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprise': (255, 153, 255),
            'neutral': (255, 255, 255),
        }

        # State Variables
        self.frame = None
        self.results = []
        self.emotion_history = {}
        self.lock = threading.Lock()
        self.running = True

        # Start Background Processing Thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()

    def _processing_loop(self):
        """Background Thread for Emotion Analysis"""
        while self.running:
            if self.frame is not None:
                try:
                    # Convert frame to RGB for DeepFace
                    rgb_frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                    # Detect faces with MTCNN
                    detections = self.DETECTOR.detect_faces(rgb_frame)
                    processed_results = []

                    for det in detections:
                        x, y, w, h = det['box']
                        cropped_face = rgb_frame[y:y+h, x:x+w]

                        # Analyze Emotion using DeepFace
                        analysis = DeepFace.analyze(
                            img_path=cropped_face,
                            actions=['emotion'],
                            enforce_detection=False,
                            detector_backend='mtcnn',
                            silent=True
                        )

                        if isinstance(analysis, list):
                            analysis = analysis[0]  # Extract first result

                        dominant_emotion = analysis['dominant_emotion']
                        confidence = analysis['emotion'][dominant_emotion]

                        if confidence >= self.MIN_CONFIDENCE:
                            processed_results.append((x, y, w, h, dominant_emotion, confidence))
                    
                    with self.lock:
                        self.results = processed_results

                except Exception as e:
                    print(f"Error: {e}")
            time.sleep(0.05)  # Prevent CPU overload

    def update_frame(self, frame):
        """Update the frame for processing"""
        self.frame = cv2.resize(frame, (640, 480))  # Resize for speed

    def get_emotions(self):
        """Retrieve processed results"""
        with self.lock:
            return self.results.copy()

    def stop(self):
        """Stop processing and cleanup"""
        self.running = False
        self.processing_thread.join()

# Initialize components
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize Emotion Detector
emotion_detector = EmotionDetector(min_confidence=70)

prediction_text = "....."
translator = Translator()
last_prediction_time = 0
selected_language = "en"
suggestor = WordSuggestor()

def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    try:
        translated = translator.translate(text, dest=target_lang)
        return translated.text
    except:
        return text

def speak_text(text):
    if text.strip():
        translated_text = translate_text(text, selected_language)
        tts_thread = threading.Thread(target=speak, args=(translated_text,))
        tts_thread.start()

def speak(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if "female" in voice.name.lower():  
            engine.setProperty('voice', voice.id)
            break
    engine.say(text)
    engine.runAndWait()

def generate_frames():
    global prediction_text, last_prediction_time
    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        # Update emotion detector with current frame
        emotion_detector.update_frame(frame.copy())
        
        # Get emotion results and draw them on frame
        emotion_results = emotion_detector.get_emotions()
        for x, y, w, h, emotion, confidence in emotion_results:
            color = emotion_detector.EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{emotion} ({confidence:.1f}%)"
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Then do sign language prediction
        current_time = time.time()
        if current_time - last_prediction_time > 2:
            last_prediction_time = current_time
            frame, prediction_text = classifier.get_prediction(frame)
            speak_text(prediction_text)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict')
def predict():
    global prediction_text
    return jsonify({'prediction': prediction_text})

@app.route('/emotion')
def emotion():
    emotion_results = emotion_detector.get_emotions()
    if emotion_results:
        # Return the first face's emotion
        return jsonify({'emotion': emotion_results[0][4]})
    return jsonify({'emotion': 'No face detected'})

@app.route('/get_suggestions')
def get_suggestions():
    prefix = request.args.get('prefix', '').lower()
    suggestions = suggestor.get_suggestions(prefix)
    corrections = []
    
    if len(suggestions) < 3:
        corrections = suggestor.get_corrections(prefix)
    
    return jsonify({
        'suggestions': suggestions,
        'corrections': corrections
    })

@app.route('/update_history', methods=['POST'])
def update_history():
    data = request.get_json()
    word = data.get('word', '').lower()
    if word:
        suggestor.update_history(word)
    return jsonify({'status': 'success'})

@app.route('/change_language', methods=['POST'])
def change_language():
    global selected_language
    selected_language = request.json.get("language", "en")
    return jsonify({'message': f'Language changed to {selected_language}'})

def on_shutdown():
    emotion_detector.stop()
    cap.release()

if __name__ == "__main__":
    import atexit
    atexit.register(on_shutdown)
    app.run(debug=True)
