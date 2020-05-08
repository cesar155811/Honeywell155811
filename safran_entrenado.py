import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
import time as tm
from keras.callbacks import EarlyStopping
import socket
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
model_2 = load_model('SAFRAN_ARANDELAS.h7')
import numpy as np
import matplotlib.pyplot as plt 
import cv2
img = cv2.imread('sin_arandela_1.jpg')





#resized=cv2.resize(img,(230,230))
resized=cv2.resize(img,(150,150))
imagen_como_matriz=np.array(resized)
imagen_como_matriz = imagen_como_matriz/255
#imagen_como_matriz_tamaño_ajustado=imagen_como_matriz.reshape(1,230,230,3)
imagen_como_matriz_tamaño_ajustado=imagen_como_matriz.reshape(1,150,150,3)
#y_pred=model.predict_proba(imagen_como_matriz_tamaño_ajustado)
y_pred=model_2.predict_proba(imagen_como_matriz_tamaño_ajustado)
print(y_pred)


# if y_pred.item(0)==1:
#     print('ROJO')
#     cv2.putText(resized,'ROJO',(100,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
#     cv2.imshow('image',resized)
#     plt.imshow(resized)
# if y_pred.item(1)==1:
#     print('ROJO Y VERDE')
#     cv2.putText(resized,'ROJO y VERDE',(100,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
#     cv2.imshow('image',resized)
#     plt.imshow(resized)
# if y_pred.item(2)==1:
#     print('VERDE')
#     cv2.putText(resized,'VERDE',(100,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
#     cv2.imshow('image',resized)
#     plt.imshow(resized)
# if y_pred.item(3)==1:
#     print('SIN COLOR')
#     cv2.putText(resized,'SIN COLOR',(100,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
#     cv2.imshow('image',resized)
#     plt.imshow(resized)
    
       
if y_pred.item(0)==1:
    print('CON ARANDELA')
    print(y_pred)
    cv2.putText(resized,'CON ARANDELA',(5,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    cv2.imshow('image',resized)
    plt.imshow(resized)
if y_pred.item(1)==1:
    print('SIN ARANDELA')
    print(y_pred)
    cv2.putText(resized,'SIN ARANDELA',(5,30),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),1)
    cv2.imshow('image',resized)
    plt.imshow(resized)


k = cv2.waitKey(0)
if k == 27:        
    cv2.destroyAllWindows()
elif k == ord('s'): 
    cv2.destroyAllWindows()