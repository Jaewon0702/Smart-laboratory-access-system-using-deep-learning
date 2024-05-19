# face_recognition
#32193430 Jaewon Lee Dankook University  32193430@dankook.ac.kr
import face_recognition
import cv2
import camera
import os
import numpy as np
import time
import object_recog
import matplotlib.pyplot as plt
import pandas as pd
import datetime

Student_Data = {
    'Jaewon' : ['32193430', 'Jaewon Lee'],
    'Jongbum' : ['32183512', 'Jongbum Lee'],
    'Sungbin' : ['32182010', 'Sungbin Bae'],
    'Younghoo': ['32183337', 'Younghoo Lee'],
    'Youngseung': ['32190175', 'Youngseung Kwak'],
    "Unknown" : ['NaN', "Unknown"]
}
global Last_name
Last_name = 'Unknown'

class FaceRecog():
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.camera = camera.VideoCamera()

        self.known_face_encodings = []
        self.known_face_names = []

        # Load sample pictures and learn how to recognize it.
        # Get user name from file
        dirname = 'knowns'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        # Initialize some variables
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        self.expected_names = []
        self.last_process_time = time.time()

    def __del__(self):
        del self.camera

    def get_frame(self):
        # Set time before model inference
        current_time = time.time()

        # Grab a single frame of video
        frame = self.camera.get_frame()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]


        # Only process every other frame of video to save time
        if current_time - self.last_process_time >= 2:
        #if process_this_frame:
            self.last_process_time = current_time
            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                min_value = min(distances)

                # tolerance: How much distance between faces to consider it a match. Lower is more strict.
                # 0.6 is typical best performance.
                name = "Unknown" # if min_value > 0.6, it's Unknown person.
                # 정확도를 높이는 법 3초동안 인식해서 제일 많이 나온 걸로 ㄱㄱ
                if min_value < 0.45:
                    index = np.argmin(distances)
                    name = self.known_face_names[index]
                self.face_names.append(name)

                # 마지막으로 인식한 이름 저장
                global Last_name
                Last_name = name
        #self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

def screenshot(frame):
    # Save Screen shot
    now = datetime.datetime.now()
    cv2.imwrite('captured/captured_%s.png' % now.strftime("%Y%m%d_%H%M%S"), frame)

    # Show Screen shot
    plt.figure(figsize=(16, 16))
    plt.imshow(frame[:, :, ::-1])
    plt.axis("off")
    plt.show()

def AddData(name, worn):
    now = datetime.datetime.now()
    df = pd.read_csv("Records/Access_Management_ledge.CSV")
    df.loc[len(df)] = Student_Data[name]+[now.strftime("%Y-%m-%d %H:%M:%S"), worn]
    print(df.head())
    df.to_csv("Records/Access_Management_ledge.CSV",index = False, mode = 'w')

if __name__ == '__main__':
    startObjRecog = False
    TEXT_PROMPT = "A person with a lab coat, safety glasses, mask."

    face_recog = FaceRecog()
    print(face_recog.known_face_names)

    start_time = 0
    obj_RecogStart = False
    screenshot_waitTime = 10

    while True:
        frame = face_recog.get_frame()
        # show the frame
        current_time = time.time()
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # 특정 버튼을 눌렀을 때(ex c) 얼굴인식 -> 실험 보호구 인식 -> 얼굴 인식 계속 스위칭이 되도록 하자..!
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        # if the 'c' was pressed, start Object recognition
        if key == ord("c"):
            obj_RecogStart = True
            start_time = time.time()
        # key를 누른지 10초가 지나면 object recognition 시작
        if(obj_RecogStart and time.time() - start_time >= screenshot_waitTime):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = object_recog.inference(frame, TEXT_PROMPT)
            screenshot(frame)
            AddData(Last_name, object_recog.WornSafetyGear)
            obj_RecogStart = False

    # do a bit of cleanup
    cv2.destroyAllWindows()
    print('finish')
