import cv2
#import OpenCv

cascPath = "haarcascade_frontalface_default.xml"
eyePath = "haarcascade_eye.xml"
smilePath = "haarcascade_smile.xml"

#Assign Haarcascade path to variables
faceCascade = cv2.CascadeClassifier(cascPath)
eyeCascade = cv2.CascadeClassifier(eyePath)
smileCascade = cv2.CascadeClassifier(smilePath)

#Classify haarcascade dataset using CascadeClassifier

font = cv2.FONT_HERSHEY_SIMPLEX
#Assign Font
video_capture = cv2.VideoCapture(0)
#take data from webcam as input using VideoCapture()

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#Change color to grayscale

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(200, 200),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
#detect face using Cascade

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.putText(frame,'Face',(x, y), font, 2,(255,0,0),5)

        smile = smileCascade.detectMultiScale(
             roi_gray,
             scaleFactor= 1.16,
             minNeighbors=35,
             minSize=(25, 25),
             flags=cv2.CASCADE_SCALE_IMAGE      
             )
#Detect smile using cascade 
                                            

        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (5, 255, 5), 1)#Coordinates of smile 
            cv2.putText(frame,'Number Of Smiles : ' + str(len(smile)),(20,20), font, 1, (0, 0, 255), 2)
            
        eyes = eyeCascade.detectMultiScale(roi_gray)
#detect eyes using cascade
             
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)#coordinates for eyes
            cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)  #to count no of faces 

            cv2.putText(frame,'Number Of Eyes : ' + str(len(eyes)),(20,20), font, 1,(0, 255, 0), 2)  # to count no of eyes

             

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
       break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
