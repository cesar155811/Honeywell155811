

import numpy as np
import matplotlib.pyplot as plt
import keras
import os
import cv2
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils

DATADIR="//mx60nt001/SHARED/COMMON/SISTEMA DE VISION EMPAQUE\Carpeta compartida\Base de datos, imagenes de 2 piezas mediana y pequenia mas background/nuevo/ENTRENAMIENTO"
CATEGORIES=['3033053-7','3104343-751','3616967-1','3822400-5','3822523-003','30600371-1','70721597-1','Background','LH70027-01']

DATAVAL="//mx60nt001/SHARED/COMMON/SISTEMA DE VISION EMPAQUE\Carpeta compartida\Base de datos, imagenes de 2 piezas mediana y pequenia mas background/nuevo/VALIDACION"
CATEGORIES_VAL=['3033053-7','3104343-751','3616967-1','3822400-5','3822523-003','30600371-1','70721597-1','Background','LH70027-01']
     
IMG_SIZE=300
number_of_classes=9

training_data=[]
validation_data=[]

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
                normalized=new_array/255
                training_data.append([normalized,class_num])
            except Exception as e:
                pass
            
def create_validation_data():
    for category in CATEGORIES_VAL:
        path=os.path.join(DATAVAL,category)
        class_num=CATEGORIES_VAL.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
               
                validation_data.append([new_array,class_num])
            except Exception as e:
                pass
            
create_training_data()
create_validation_data()
random.shuffle(training_data)


X=[]
Y=[]

W=[]
V=[]


for features,label in training_data:
    X.append(features)
    Y.append(label)
    
    
for features, label in validation_data:
    W.append(features)
    V.append(label)
    
    
X=np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
W=np.array(W).reshape(-1,IMG_SIZE,IMG_SIZE,1)


print(X.shape)
#print(Y)
print(W.shape)
#print(V)
    
#Algoritmo
def leNet_model():
    model=Sequential()
    model.add(Conv2D(16,(3,3),input_shape=(300,300,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(number_of_classes,activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    return model

model=leNet_model()
print(model.summary())
#entrenamiento
model.fit(X,Y,validation_data=(W,V),epochs=10,batch_size=128,verbose=1,shuffle=1)

model.save('CNN_IMPELLERS_10.h7')

metrics=model.evaluate(W,V)
print("Metrics(test loss & Test Accuracy): ")
print(metrics)


    






    
