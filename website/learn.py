from flask import Blueprint, Flask, render_template, request, flash, redirect, url_for, Response
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db   ##means from __init__.py import db
from flask_login import login_user, login_required, logout_user, current_user
import pickle
import mediapipe as mp
import numpy as np
import cv2
import time
from random import random
from flask import jsonify


learn = Blueprint('learn', __name__)

# letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
letters = ['Y', 'O', 'U']
words = ['LOVE', 'MOON', 'STAR', 'SUN', 'FIRE', 
         'RAIN', 'SNOW', 'WIND', 'TREE', 'BIRD', 
         'FISH', 'DOOR', 'BOOK', 'COOK', 'FOOD', 
         'SOUP', 'CAKE', 'MILK', 'EGG', 'SALT', 
         'RICE', 'BEAN', 'CORN', 'HERB', 'MEAT', 
         'PORK', 'BEEF', 'LAMB', 'KALE', 'KIWI', 
         'LIME', 'LEAF', 'CLAY', 'WOOD', 'IRON', 
         'GOLD', 'ROCK', 'HILL', 'LAKE', 'POND', 
         'RIVER', 'CITY', 'TOWN', 'ROAD', 'PATH', 
         'PARK', 'SHOP', 'PLAY', 'READ', 'DRAW']

easy = 1 
cap = cv2.VideoCapture(0, cv2.CAP_ANY)
easy_word_user = ''
eraser = 0
easy_word = words[int(random()*len(words))].upper()
easy_word_index = 0
location = 0
letter_help = 0
curr_time = 0
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

@learn.route('/learn', methods=['GET', 'POST'])
def live_translate():
    global easy_word
    return render_template("stage1.html", user=current_user)

@learn.route('/req',methods=['POST','GET'])
def mode():
    global    easy
    if request.method == 'POST':
        if request.form.get('easy') == 'Easy':
            easy= not easy
            # medium, hard, free =  0, 0, 0
        # elif  request.form.get('medium') == 'Medium':
        #     medium=not medium
        #     easy, hard, free =  0, 0, 0
        # elif  request.form.get('hard') == 'Hard':
        #     hard=not hard
        #     easy, medium, free =  0, 0, 0
        # elif  request.form.get('free') == 'Freestyle':
        #     free=not free  
        #     easy = 0
        #     medium = 0
        #     hard = 0
        '''elif  request.form.get('switch') == 'Stop/Start':
            
            if switch:
                switch=0
                cap.release()
                cv2.destroyAllWindows()
                
            else:
                cap = cv2.VideoCapture(camera_max())
                switch=1'''
                          
                 
    elif request.method=='GET':
        return render_template('stage1.html', user=current_user)
    return render_template('stage1.html', user=current_user)

@learn.route('/new_word', methods=['GET'])
def new_word():
    global easy_word, easy_word_index, easy_word_user
    import random
    easy_word = random.choice(words).upper()
    easy_word_index = 0  # Reset index
    easy_word_user = ''  # Reset the user's progress on the current word
    return jsonify({'new_word': easy_word})


@learn.route('/learn_video_feed')
def learn_video_feed():
    return Response(learn_sign_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

def learn_sign_frame():  # generate frame by frame from camera
    global easy, cap
    while True:
        success, frame = cap.read() 
        if success:
            if(easy):             
                frame = easy_mode(frame)
            # elif(medium):                
            #     frame = medium_mode(frame)
   
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

def easy_mode(frame):
    global model, words, easy_word, easy_word_user, easy_word_index
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    labels_dict = ['Y', 'O', 'U']
    while True:
        ret, frame = cap.read()

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_4)
            cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
        except Exception as e:
            print(e)
        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            if num_hands > 1:
                cv2.waitKey(5)
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
                probabilities = model.predict_proba([np.asarray(data_aux)])
                prediction = np.argmax(probabilities, axis=1)  # This assumes your model's predict_proba returns a list of probabilities for each class
                predicted_class = model.predict([np.asarray(data_aux)])  # Assuming single prediction
                prediction_probability = np.max(probabilities, axis=1)[0]  # Get the max probability
                prediction_probability_percent = prediction_probability * 100
                
                # Draw bounding box and prediction on frame for the current hand
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                prediction_text = f'{str(predicted_class)} ({prediction_probability_percent:.2f}%)'
                cv2.putText(frame, prediction_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                
                if str(predicted_class[0]).upper() == str(easy_word[easy_word_index]):
                    
                    # Mark the letter in green on the screen
                    easy_word_user += str(predicted_class[0]).upper()
                    
                    easy_word_index += 1
                    # Check if the word is completed
                    if easy_word_index >= len(easy_word):
                        cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
                        cv2.waitKey(15)
                        # Select a new word and reset the index
                        easy_word = words[np.random.randint(len(words))].upper()
                        easy_word_index = 0
                        easy_word_user = ''
                


        return frame

# def easy_mode(frame):
#     global cap, easy_word_user, easy_word, easy_word_index, curr_time, location, letter_help

    
#     def mediapipe_detection(image, model):
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
#         results = model.process(image)                 # Make prediction
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
#         return image, results

#     def get_landmark_dist_test(results, x, y):
#         hand_array = []
#         wrist_pos = results.multi_hand_landmarks[0].landmark[0]
#         for result in results.multi_hand_landmarks[0].landmark:
#             mp_drawing.draw_landmarks(
#                     frame,  # image to draw
#                     result,  # model output
#                     mp_hands.HAND_CONNECTIONS,  # hand connections
#                     mp_drawing_styles.get_default_hand_landmarks_style(),
#                     mp_drawing_styles.get_default_hand_connections_style())
#         for result in results.multi_hand_landmarks[0].landmark:
#             hand_array.append((result.x-wrist_pos.x) * (width/x))
#             hand_array.append((result.y-wrist_pos.y) * (height/y))
#         return(hand_array[2:])


    #Main function
    #cap = cv2.VideoCapture(cam_max)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    model_dict = pickle.load(open('./model.p', 'rb'))
    clf = model_dict['model']

    start_time = time.time()

    # Set mediapipe model
    mp_hands = mp.solutions.hands # Hands model
    with mp_hands.Hands(min_detection_confidence=0.1, min_tracking_confidence=0.1, max_num_hands=1) as hands:
        while cap.isOpened():
            # Read feed
            #ret, frame = cap.read()

            try:
                cv2.putText(frame, easy_word, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_4)
                cv2.putText(frame, easy_word_user, (int(width*0.05), int(height*0.95)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_4)
                
            except Exception as e:
                print(e)

            # Make detections
            image, results = mediapipe_detection(frame, hands)

            # letter_help = cv2.resize(cv2.imread('easy_mode_letters/{}.png'.format(easy_word[easy_word_index].lower())), (0,0), fx=0.2, fy=0.2)

            #Find bounding box of hand
            
            if results.multi_hand_landmarks:
                x = [None,None]
                y=[None,None]
                for result in results.multi_hand_landmarks[0].landmark:
                    if x[0] is None or result.x < x[0]: x[0] = result.x
                    if x[1] is None or result.x > x[1]: x[1] = result.x

                    if y[0] is None or result.y < y[0]: y[0] = result.y
                    if y[1] is None or result.y > y[1]: y[1] = result.y

                for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )


                if curr_time < round((time.time() - start_time)/3,1) and x[0] is not None:
                        curr_time = round((time.time() - start_time)/3,1)
                        try:
                            test_image = get_landmark_dist_test(results, x[1]-x[0], y[1]-y[0])
                            test_pred = np.argmax(clf.predict_proba(np.array([test_image])))
                            test_probs = clf.predict_proba(np.array([test_image]))[0]
                            print("Predicted:",letters[test_pred], ", pred prob:", max(test_probs), ", current index:", easy_word_index, ", current time:", curr_time)
                            if max(test_probs) >= 0.8 or (max(test_probs) >= 0.6 and letters[test_pred] in ['p','r','u','v']):
                                pred_letter = letters[test_pred].upper()
                                cv2.putText(frame, pred_letter, (x[1], y[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and (easy_word_index == 0 or easy_word[easy_word_index] != easy_word[easy_word_index - 1]):
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x
                                if easy_word_index < len(easy_word) and pred_letter == easy_word[easy_word_index] and easy_word_index > 0 and easy_word[easy_word_index] == easy_word[easy_word_index - 1] and abs(location - results.multi_hand_landmarks[0].landmark[0].x) > 0.1:
                                    easy_word_user += pred_letter
                                    easy_word_index += 1
                                    location = results.multi_hand_landmarks[0].landmark[0].x

                            if easy_word_user == easy_word:
                                time.sleep(0.5)
                                easy_word = words[int(random()*len(words))].upper()
                                print(easy_word)
                                easy_word_index = 0
                                easy_word_user = ''

                        except Exception as e:
                            print(e)

            # Show letter helper
            # frame[5:5+letter_help.shape[0],width-5-letter_help.shape[1]:width-5] = letter_help

            return frame
            
    return frame  