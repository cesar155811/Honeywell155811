
import numpy as np
import keras
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

directorioPrincipal="C:/Users/H155811/Desktop/COSAS_NEGRAS"
    
hsize=100
vsize=100

def create_training_data(directorioPrincipal,categorias,hsize,vsize):
  
    valores=0    
    for category in categorias:
        path=os.path.join(directorioPrincipal,category)
        class_num=categorias.index(category)         
        valor=len(os.listdir(path))    
        valores=valores+valor  
    trainingImageSet= np.zeros((valores,vsize,hsize,3))
    trainingLabelSet= np.zeros((valores,1))
    
    i=0      
    for category in categorias:
        path=os.path.join(directorioPrincipal,category)
        class_num=categorias.index(category) 
        
        for img in os.listdir(path):
                file_extension = os.path.splitext(os.path.join(path,img))
                
                if file_extension[1]==permittedExtension:
                  img_array=cv2.imread(os.path.join(path,img))               
                  img_array=cv2.resize(img_array,(hsize,vsize))
                  img_array=img_array/np.amax(img_array)
                  trainingImageSet[i,:,:,:]=img_array
                  trainingLabelSet[i,:]=class_num
                  i=i+1

    return  trainingImageSet,trainingLabelSet       

permittedExtension=".jpg"

categorias=os.listdir(directorioPrincipal)
if "Thumbs.db" in categorias:
    categorias.remove("Thumbs.db")

number_of_classes=len(categorias)

trainingImageSet,trainingLabelSet=create_training_data(directorioPrincipal,categorias,hsize,vsize)
y_train = keras.utils.to_categorical(trainingLabelSet, num_classes=number_of_classes)

X_train, X_test, Y_train, Y_test = train_test_split(trainingImageSet, y_train, test_size=0.30, random_state=101)

train_test_split(y_train, shuffle=True)


def leNet_model():
    model=Sequential()
    model.add(Conv2D(32,(2,2),input_shape=(vsize,hsize,3),strides = (1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(2,2),strides = (1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(5000,activation='relu'))
    model.add(Dense(number_of_classes,activation='softmax'))   
    model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    return model

model=leNet_model()
print(model.summary())

datagen = ImageDataGenerator(
    rotation_range=45,
    zoom_range=-0.2,
    shear_range=0.2)

datagen.fit(X_train)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),steps_per_epoch=len(X_train)/32, epochs=100)

metrics = model.evaluate(X_test, Y_test, batch_size=32)

model.save('CNN_SAFRAN_COSANEGRA.h7')

print("Metrics(test loss & Test Accuracy): ")
print(metrics)
