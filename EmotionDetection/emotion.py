import cv2
from deepface import DeepFace
import serial

ser = serial.Serial('/dev/tty.usbserial-210', 9600)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
    	#rectangles around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']

        if emotion == 'angry':
            serial_emotion = 'r'
        elif emotion == 'sad':
            serial_emotion = 'b'
        elif emotion == 'happy':
            serial_emotion = 'y'
        else:
            serial_emotion = 'w'

        print(f"{serial_emotion}")

        ser.write(serial_emotion.encode())  # Send the detecqted emotion via serial

    cv2.imshow('Video', frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
ser.close()
