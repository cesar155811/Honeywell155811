
import cv2
from keras.models import load_model
model = load_model('SAFRAN_SOLO_COLORES_6.h7')
import numpy as np
import matplotlib.pyplot as plt 

img = cv2.imread('3.jpg')
img2 = np.copy(img)

count = 0

# crop_1 = img[0:160,0:640]

X=[]

    
for i in range(0,480,160):
    for j in range(0,640,160):
        n = img2[i:160+i,j:j+160]
        cv2.imwrite('frame%d.jpg' % count,n)
        count += 1
        X.append(n)
         
mat_1 = np.concatenate((X[0],X[1],X[2],X[3]), axis=1)
mat_2 = np.concatenate((X[4],X[5],X[6],X[7]), axis=1)
mat_3 = np.concatenate((X[8],X[9],X[10],X[11]), axis=1)

mat_completa = np.concatenate((mat_1,mat_2,mat_3), axis = 0)

for imagen in X:
    p=cv2.resize(imagen,(230,230))
    imagen_como_matriz_tamaño_ajustado=p.reshape(1,230,230,3)
    imagen_como_matriz_tamaño_ajustado = imagen_como_matriz_tamaño_ajustado/255
    y_pred=model.predict(imagen_como_matriz_tamaño_ajustado)
    print(y_pred)
    if y_pred.item(0)==1:
        print('ROJO')
    if y_pred.item(1)==1:
        print('ROJO Y VERDE')
    if y_pred.item(2)==1:
        print('VERDE')
    if y_pred.item(3)==1:
        print('SIN COLOR')


cv2.imshow('img',mat_completa)

k = cv2.waitKey(0)
if k == 27:        
    cv2.destroyAllWindows()
elif k == ord('s'): 
    cv2.destroyAllWindows()
    
    