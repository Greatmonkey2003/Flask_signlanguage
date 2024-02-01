from flask import Flask, render_template, Response, redirect, request, send_file
import cv2
import mediapipe as mp
import joblib
import numpy as np
import tensorflow as tf
import keras
#import pyttsx3
import warnings
import base64
import io

# Ignore the specific UserWarning about feature names
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

camera_running = False
clf = joblib.load('american_svm_model (1).pkl')

model_path = "Emotion_Recognition (1).h5"
hand_sign_model = keras.models.load_model(model_path)
# For webcam input:
hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

detected_word = ''
word_in_progress = False

def data_clean(landmark):
    data = landmark[0]

    try:
        data = str(data)
        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = [i for i in data if i not in garbage]

        clean = [float(i.strip()[2:]) for i in without_garbage]

        return [clean]

    except:
        return np.zeros([1, 63], dtype=int)[0]

def speak_letter(letter):
    pass
    # engine = pyttsx3.init()
    # engine.say(f"{letter}")
    # engine.runAndWait()

def generate_frames():
    hands = mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()

        image = cv2.flip(image, 1)

        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cleaned_landmark = data_clean(results.multi_hand_landmarks)

            if cleaned_landmark:
                #clf = joblib.load('american_svm_model (1).pkl')
                y_pred = clf.predict(cleaned_landmark)
                detected_letter = str(y_pred[0])

                # Additional logic for handling detected letters
                # if detected_letter == 'B' and detected_word:
                #     detected_word = detected_word[:-1]  # Remove the last letter
                # else:
                #     detected_word += detected_letter

                image = cv2.putText(image, detected_letter, (50, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', image)
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    hands.close()
    cap.release()

# Import statements...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/open_camera')
def open_camera():
    print(camera_running)
    # Additional logic for opening camera (if needed)
    return render_template('camera.html')


@app.route('/video_feed')
def video_feed():
    print(camera_running)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_camera')
def stop_camera():
    global camera_running
    camera_running = True

    try:
        print(camera_running)
        if camera_running:
            print("hurray if ",camera_running)
            # Additional logic for stopping camera (if needed)
            hands.close()
            cap.release()
            camera_running = False
    except Exception as e:
        # Handle the exception, log it, or perform other necessary actions
        print(f"Error stopping camera: {e}")

    return redirect('/')

def pred_and_plot(file_storage):
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    # Read the image from the FileStorage object
    img = load_and_prep_image(file_storage)

    # Convert the image to BGR format for compatibility with cv2.imencode
    img_bgr = cv2.cvtColor(img.numpy(), cv2.COLOR_GRAY2BGR)

    # Convert the image to base64 for displaying in HTML
    _, img_bytes = cv2.imencode('.jpg', img_bgr)
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Make a prediction
    pred = hand_sign_model.predict(tf.expand_dims(img, axis=0))
    print(pred)
    if len(pred[0]) > 1:  # Check for multi-class
        pred_class = class_names[pred.argmax()]  # If more than one output, take the max
    else:
        pred_class = class_names[int(tf.round(pred)[0][0])]  # If only one output, round

    return pred_class, img_base64


def load_and_prep_image(file_storage, img_shape=48):
    # Read the image from the FileStorage object
    img = tf.io.decode_image(file_storage.read(), channels=3)  # Ensure 3 channels (RGB)
    img = tf.image.rgb_to_grayscale(img)  # Convert to grayscale
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.0  # Normalize to [0, 1]
    return img

@app.route('/pick_from_gallery', methods=['GET', 'POST'])
def pick_from_gallery():
    prediction = None
    uploaded_image_base64 = None

    if request.method == 'POST':
        # Get the uploaded image file
        uploaded_file = request.files['image']

        if uploaded_file:
            # Make a prediction and convert the image to base64
            prediction, uploaded_image_base64 = pred_and_plot(uploaded_file)
            prediction = f"The predicted hand sign is {prediction}"
    # Render the form with the uploaded image and prediction result
    return render_template('pick_from_gallery.html', prediction=prediction, uploaded_image_base64=uploaded_image_base64)


#if __name__ == "__main__":
#    app.run(debug=False,host='0.0.0.0')

