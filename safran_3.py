
import cv2
import numpy as np

img = cv2.imread('3.jpg')
img2 = np.copy(img)

count = 0

crop_1 = img[0:160,0:640]

X=[]

# for i in range(0,480,160):
#     m=img2[i:160+i,0:640]
#     cv2.imwrite('frame_x%d.jpg' % count_x,m)
#     count_x += 1
#     X.append(m)

# for j in range(0,640,200):
#     n = img2[0:480,j:j+200]
#     cv2.imwrite('frame_y%d.jpg' % count_y,n)
#     count_y += 1
#     Y.append(n)
    
for i in range(0,480,160):
    for j in range(0,640,200):
        n = img2[i:160+i,j:j+200]
        cv2.imwrite('frame%d.jpg' % count,n)
        count += 1
        X.append(n)
         
mat_1 = np.concatenate((X[0],X[1],X[2],X[3]), axis=1)
mat_2 = np.concatenate((X[4],X[5],X[6],X[7]), axis=1)
mat_3 = np.concatenate((X[8],X[9],X[10],X[11]), axis=1)

mat_completa = np.concatenate((mat_1,mat_2,mat_3), axis = 0)


cv2.imshow('img',mat_completa)

k = cv2.waitKey(0)
if k == 27:        
    cv2.destroyAllWindows()
elif k == ord('s'): 
    cv2.destroyAllWindows()