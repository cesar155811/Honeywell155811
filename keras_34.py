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

DATADIR="//mx60nt001/SHARED/COMMON/SISTEMA DE VISION EMPAQUE/Carpeta compartida/base_de_datos_super_nueva/Entrenamiento"
CATEGORIES=['698192-4','3033053-7','3033207-1','3616967-1','3822249-4','3822400-5','3822523-003','70721597-1','fondo','LH70027-01']

DATAVAL="//mx60nt001/SHARED/COMMON/SISTEMA DE VISION EMPAQUE/Carpeta compartida/base_de_datos_super_nueva/validacion"
CATEGORIES_VAL=['698192-4','3033053-7','3033207-1','3616967-1','3822249-4','3822400-5','3822523-003','70721597-1','fondo','LH70027-01']
     
#IMG_SIZE=300
number_of_classes=10

training_data=[]
validation_data=[]

def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(100,100))
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
                img_array=cv2.imread(os.path.join(path,img))
                new_array=cv2.resize(img_array,(100,100))
               
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
    
    
X=np.array(X).reshape(-1,100,100,3)
W=np.array(W).reshape(-1,100,100,3)


print(X.shape)
#print(Y)
print(W.shape)
#print(V)
    
#Algoritmo
def leNet_model():
    model=Sequential()
    model.add(Conv2D(16,(2,2),input_shape=(100,100,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(256,(2,2),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(5000,activation='relu'))
    model.add(Dense(5000,activation='relu'))
    model.add(Dense(number_of_classes,activation='softmax'))
    
    model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    return model

model=leNet_model()
print(model.summary())
#entrenamiento
model.fit(X,Y,validation_data=(W,V),epochs=45,batch_size=64,verbose=1,shuffle=1)

model.save('CNN_IMPELLERS_21.h7')

metrics=model.evaluate(W,V)
print("Metrics(test loss & Test Accuracy): ")
print(metrics)


    






    
