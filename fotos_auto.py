
import cv2
import time

vidcap = cv2.VideoCapture(0)

success,image=vidcap.read()

count = 899

success = True

while success:
    
    success, image = vidcap.read()
    
    crop = image[0:480,100:560]
    
    cv2.imwrite('frame%d.jpg' % count, crop)
    
    count += 1
    
    cv2.imshow("resize", crop)
    
    time.sleep(1)
    
    if cv2.waitKey(1) & 0xFF  == ord('q'):
        
        break
    
vidcap.release()
cv2.destroyAllWindows()