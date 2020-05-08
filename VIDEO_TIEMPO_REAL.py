
import cv2
import numpy as np
 
capture = cv2.VideoCapture(0)
img_counter = 0 

while(True):
     
    ret, frame = capture.read()
    
    frame = cv2.resize(frame,(640,480))
    
    frame_2 = np.copy(frame)
    
    for i in range(0,480,160):
        cv2.line(frame,(0,i),(640,i),(0,255,0),2)
        for j in range(0,640,160):
            cv2.line(frame,(j,0),(j,480),(0,255,0),2)
     
    cv2.imshow('video', frame)
     

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame_2)
        print("{} written!".format(img_name))
        img_counter += 1

capture.release()
cv2.destroyAllWindows()




