
import cv2

panda_1 = cv2.imread('safran_ejemplo.jpg')

# panda_1=cv2.resize(panda,(640,480))

cv2.rectangle(panda_1,(240,240),(290,290),(0,255,0),2)
cv2.rectangle(panda_1,(410,250),(460,300),(0,255,0),2)
cv2.rectangle(panda_1,(600,250),(650,300),(0,255,0),2)


cv2.imwrite('safran_cuadro.png',panda_1)

cv2.imshow('fondo',panda_1)

k = cv2.waitKey(0)
if k == 27:        
    cv2.destroyAllWindows()
elif k == ord('s'): 
    cv2.destroyAllWindows()


