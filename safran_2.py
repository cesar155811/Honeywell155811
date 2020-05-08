import cv2
from PIL import Image, ImageDraw 

img = cv2.imread('3.jpg')

crop_1 = img[0:240,0:200]
crop_2 = img[000:240,200:400]
crop_3 = img[00:240,400:640]
crop_4 = img[200:480,0:200]
crop_5 = img[200:480,200:400]
crop_6 = img[200:480,400:640]

crop_7 = img[260:380,240:360]
crop_8 = img[100:300,100:300]

largo = img.shape[0]/4
ancho = img.shape[1]/4

# cv2.line(img,(0,120),(640,120),(0,255,0),3)
# cv2.line(img,(0,240),(640,240),(0,255,0),3)
# cv2.line(img,(0,360),(640,360),(0,255,0),3)

# cv2.line(img,(130,0),(130,480),(0,255,0),3)
# cv2.line(img,(260,0),(260,480),(0,255,0),3)
# cv2.line(img,(380,0),(380,480),(0,255,0),3)
# cv2.line(img,(500,0),(500,480),(0,255,0),3)

cv2.rectangle(img,(0,160),(200,320),(0,255,0),2)

cv2.rectangle(img,(200,160),(400,320),(0,255,0),2)

cv2.rectangle(img,(400,160),(600,320),(0,255,0),2)

# cv2.rectangle(img,(450,260),(550,350),(0,255,0),2)

cv2.putText(img,'ROJO Y VERDE',(20,150),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
cv2.putText(img,'SIN MARCA',(210,150),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
cv2.putText(img,'ROJO',(410,150),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
# cv2.putText(img,'CON ARANDELA',(440,250),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    
# for x in range(0,480,160):
#     cv2.rectangle(img,(0,x),(640,x),(0,255,0),2)
# for x in range(0,640,200):
#     cv2.rectangle(img,(x,0),(x,480),(0,255,0),2)

cv2.imshow('img',img)

k = cv2.waitKey(0)
if k == 27:        
    cv2.destroyAllWindows()
elif k == ord('s'): 
    cv2.imwrite('messigray.png',img)
    cv2.destroyAllWindows()