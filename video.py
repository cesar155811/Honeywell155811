
import cv2
import numpy as np
 
capture = cv2.VideoCapture(0)
panda_1 = cv2.imread('safran_cuadro.png')

while(True):
     
    ret, frame = capture.read()
    
    frame = cv2.resize(frame,(961,540))
    
    cv2.rectangle(frame,(240,240),(290,290),(0,255,0),2)
    cv2.rectangle(frame,(410,250),(460,300),(0,255,0),2)
    cv2.rectangle(frame,(600,250),(650,300),(0,255,0),2)
       
    frame_final = cv2.addWeighted(frame,1.0,panda_1,0.5,0)
    
    res = cv2.matchTemplate(frame,panda_1,cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.4
    
    if res >= threshold:
        cv2.imwrite('foto.png',frame)
        print('foto guardada')
     
    cv2.imshow('video', frame_final)
     

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break
    
capture.release()
cv2.destroyAllWindows()
