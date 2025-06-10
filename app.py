import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
from keras.models import load_model
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

model = load_model('model/6_class_emotion_detector_V2.h5')
classifier = cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')
int2emotions = {0: 'angry', 1: 'fear', 2: 'happy', 3: 'neutral', 4: 'sad', 5: 'surprise'}

def detect_emotion_in_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, 1.3, 4)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1)
        emotion = int2emotions[np.argmax(model.predict(face, verbose=0))]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (172, 42, 251), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (106, 40, 243), 2)
    return frame

@app.route('/')
def index():
    return render_template('index.html', default_view="upload")

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return redirect(url_for('index'))

    file = request.files['image']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    image = detect_emotion_in_frame(image)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + filename)
    cv2.imwrite(result_path, image)

    return render_template('index.html', result_image=result_path, original_image=filepath, default_view="upload")

def gen():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_emotion_in_frame(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
