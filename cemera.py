import cv2
import time
import datetime
import send_email
import threading
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Camera could not be opened.")
    exit()
size=(int(camera.get(3)), int(camera.get(4)))
char_code=cv2.VideoWriter_fourcc(*'mp4v')
out=cv2.VideoWriter("video.mp4", char_code, 20, size)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

detected=False
after_detect_seconds=8
time_after_detection=None
time_started=False

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("Warning: Failed to capture frame. Retrying...")
        continue
    gray_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray_frame, 1.2, 5)
    bodies=body_cascade.detectMultiScale(gray_frame, 1.2, 5)
    
    if(len(faces)+len(bodies))>0:
        if detected:
            time_started=False
        else:
            detected=True
            current=datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out=cv2.VideoWriter(f"{current}.mp4", char_code, 20, size)
            print("started recording!!", time.time())
    elif detected:
        if time_started:
            if (time.time()-time_after_detection>=after_detect_seconds):
                detected=False
                time_started=False
                out.release()
                threading.Thread(target=send_email.dispatch_email, args=(f"{current}.mp4", )).start()
                print("stoped recording!!", time.time())
        else:
            time_started=True
            time_after_detection=time.time()
    if detected:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        out.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
out.release()
camera.release()
cv2.destroyAllWindows()
