import tensorflow as tf
import numpy as np
import time
import cv2
# Caution: if you are using camera in other program, please turn off it and run this code.
# Define the input size of the model
input_size = (224, 224)

# Open the web cam
cap = cv2.VideoCapture(0)  # webcam 불러오기

if not cap.isOpened():
    print("Could not open webcam")
    exit()
cap.set(cv2.CAP_PROP_BUFFERSIZE, 5)

# without this code, the webcam works!
# Load the saved model
model = tf.keras.models.load_model("face_model.h5", compile=False)
while cap.isOpened():
    # Set time before model inference
    start_time = time.time()

    # Reading frames from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame for the model
    model_frame = cv2.resize(frame, input_size, frame)
    # Expand Dimension (224, 224, 3) -> (1, 224, 224, 3) and Normalize the data
    model_frame = np.expand_dims(model_frame, axis=0) / 255.0

    # Predict
    is_obj_prob = model.predict(model_frame)[0]
    obj_detection = np.argmax(is_obj_prob)

    # Compute the model inference time
    inference_time = time.time() - start_time
    fps = 1 / inference_time
    fps_msg = "Time: {:05.1f}ms {:.1f} FPS".format(inference_time * 1000, fps)

    # Add Information on screen
    if (is_obj_prob[obj_detection] > 0.6):
        if obj_detection == 0:
            msg_mask = "JB"
        # Add Database logic(학번, 학과, name, 현재 시간 등)
        elif obj_detection == 1:
            msg_mask = "Young_Seung"
        elif obj_detection == 2:
            msg_mask = "J1"
        elif obj_detection == 3:
            msg_mask = "bin"
        elif obj_detection == 4:
            msg_mask = "Young_Who"
        elif obj_detection == 5:
            msg_mask = "Blank"
    else:
        msg_mask = "Unknown"

    # Show probability
    msg_mask += " ({:.1f})%".format(is_obj_prob[obj_detection] * 100)

    cv2.putText(frame, fps_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
    cv2.putText(frame, msg_mask, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    # Show the result and frame
    cv2.imshow('face mask detection', frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
