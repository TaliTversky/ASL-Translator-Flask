from flask import Blueprint, render_template, request, flash, redirect, url_for, Response
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   ##means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
import cv2
import mediapipe as mp

import pickle
import numpy as np

translate = Blueprint('translate', __name__)

cap = cv2.VideoCapture(0, cv2.CAP_ANY)
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

@translate.route('/translate', methods=['GET', 'POST'])
def live_translate():
    return render_template("translate.html", user=current_user)

@translate.route('/video_feed')
def video_feed():
    return Response(sign_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def sign_frame():  # generate frame by frame from camera
    global cap
    while True:
        success, frame = cap.read() 
        if success:           
            frame = stream_translation(frame)
            # elif(medium):                
            #     frame = medium_mode(frame)
   
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

def stream_translation(frame):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    while True:
        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            if num_hands > 1:
                cv2.waitKey(15)
                cv2.putText(frame, "Only show one hand", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks for each hand
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Initialize data_aux for each hand
                data_aux = []

                # Extract landmarks for the current hand
                x_ = []
                y_ = []
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Calculate bounding box for the current hand
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Append hand landmarks to data_aux
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                # Make prediction for the current hand
                # Make prediction for the current hand
                probabilities = model.predict_proba([np.asarray(data_aux)])
                prediction = np.argmax(probabilities, axis=1)  # This assumes your model's predict_proba returns a list of probabilities for each class
                predicted_class = model.predict([np.asarray(data_aux)])  # Assuming single prediction
                prediction_probability = np.max(probabilities, axis=1)[0]  # Get the max probability
                prediction_probability_percent = prediction_probability * 100
                
                # Draw bounding box and prediction on frame for the current hand
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                prediction_text = f'{str(predicted_class)} ({prediction_probability_percent:.2f}%)'
                cv2.putText(frame, prediction_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        return frame