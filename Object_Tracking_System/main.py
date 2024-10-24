from ultralytics import YOLO
import cv2
#load yolov8
model = YOLO('yolov8n.pt')

#load video
video_path = ('./video.mp4')
cap = cv2.VideoCapture(video_path)

ret = True

while ret:
    ret, frame =  cap.read() #reading a new frame from the video
    if ret:
        #track objects
        results = model.track(frame, persist=True) #persist is used because we want YOLO to remember all the objects through all the frames seen in the video

        #plot
        # cv2.rectangle #same as below
        # cv2.putText #same as below
        frame = results[0].plot() #0 is used because we are detecting all objects and all the object tracking in only one frame

        #visualization
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xff == ord('q'):
            break
