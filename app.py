from flask import Flask, request, jsonify
from flask_mail import Mail, Message
from keras.models import load_model
import cv2, numpy as np, datetime, os
from dotenv import load_dotenv
from flask_cors import CORS

# Load env variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Email Configuration
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD")
)
mail = Mail(app)

# Emotion model and constants
model = load_model("facial_emotion_model.h5")
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
negative_emotions = {'Angry', 'Fear', 'Sad', 'Disgust'}

# Global state
emotion_logs = []
current_emotion = "Neutral"
allowed_emotions = {e: True for e in emotions}

# Function to send alert email
def send_alert_email(emotion):
    msg = Message(
        subject="🚨 Alert: Negative Emotion Detected!",
        sender=os.getenv("MAIL_USERNAME"),
        recipients=[os.getenv("PARENT_EMAIL")]
    )
    msg.body = f"A negative emotion was detected in your child: {emotion}"
    mail.send(msg)

@app.route("/")
def home():
    return "Flask API is working!"

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    global current_emotion
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    emotion = "No Face"
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        preds = model.predict(roi_gray)[0]
        emotion = emotions[np.argmax(preds)]
        break

    current_emotion = emotion
    emotion_logs.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "emotion": emotion
    })

    # Send email if emotion is negative and not allowed
    if not allowed_emotions.get(emotion, True) or emotion in negative_emotions:
        send_alert_email(emotion)

    return jsonify({"emotion": emotion})

@app.route('/emotion-logs')
def get_logs():
    return jsonify({"logs": emotion_logs[-20:]})

@app.route('/current-emotion')
def get_current():
    return jsonify({
        "current": current_emotion,
        "allowed": allowed_emotions
    })

@app.route('/update-permissions', methods=["POST"])
def update_permissions():
    global allowed_emotions
    allowed_emotions = request.json
    return jsonify({"status": "success", "allowed": allowed_emotions})

if __name__ == '__main__':
    app.run(debug=True)
