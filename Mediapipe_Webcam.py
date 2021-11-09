import cv2
import mediapipe as mp
import time
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model


cap = cv2.VideoCapture(0)
model = load_model("./data/0.9011-0.6676.hdf5", compile=False)
EMOTIONS = ["Angry", "Disgusting", "Fearful", "Happy", "Sad", "Surprise", "Neutral"]

mp_facedetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mp_facedetection.FaceDetection(0.5)

pTime = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1) # 좌우 대칭 변경
    frame_copy = frame.copy()
    frame_copy = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = faceDetection.process(frame_copy)

    if faces.detections:
        for num, detection in enumerate(faces.detections):
            box = detection.location_data.relative_bounding_box
            fh, fw, fc = frame.shape
            bbox = int(box.xmin * fw), int(box.ymin * fh), int(box.width * fw), int(box.height * fh)

            x, y, w, h = bbox
            frame_ROI = frame[y:y+h, x:x+w]
            frame_ROI = cv2.resize(frame_ROI, (48, 48))
            frame_ROI = cv2.cvtColor(frame_ROI, cv2.COLOR_BGR2GRAY)
            frame_ROI = frame_ROI.astype('float') / 255.0
            frame_ROI = img_to_array(frame_ROI)
            frame_ROI = np.expand_dims(frame_ROI, axis=0)
            result = EMOTIONS[np.argmax(model.predict(frame_ROI)[0])]

            cv2.rectangle(frame, bbox, (255, 0, 255), 2)
            cv2.putText(frame, result, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    cTime = time.time()
    fps = 1 / (cTime-pTime)
    pTime = cTime

    cv2.putText(frame, 'FPS: {}'.format(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    cv2.imshow("Web", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()