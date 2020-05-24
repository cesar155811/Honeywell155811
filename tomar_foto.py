
import cv2
import numpy as np
 
capture = cv2.VideoCapture(1)


while(True):
     
    ret, frame = capture.read()
    
    k = cv2.waitKey(1)
    if k == ord('s'): 
        cv2.imwrite('foto.png',frame)
        print('foto guardada')
     
    cv2.imshow('video', frame)
     
    if k%256 == 27:
        print("Escape hit, closing...")
        break
    
capture.release()
cv2.destroyAllWindows()
