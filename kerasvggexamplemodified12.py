# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:35:09 2019
@author: E803562
"""
# =============================================================================
# Codigo para entrenamiento de imagenes y generacion de modelo
# =============================================================================
# =============================================================================
# User Inputs
# =============================================================================
# link a la carpeta principal
#directorioPrincipal="C:/Users/E803562/Desktop/Artificial Intelligence/Caracteres"
directorioPrincipal="C:/Users/H155811/Desktop/SAFRAN/Fotos_2/ETIQUETAS/ENTRENAMIENTO"
#Tamaño de imagen
hsize=250
vsize=220
###############################################################################
###############################################################################
#Tener cuidado a partir de aqui <<<<--------------------------------
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import cv2
import time as tm
from keras.callbacks import EarlyStopping
import socket
# =============================================================================
# Functions
# =============================================================================
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
#                  
#                  print(img)
#                  print(os.path.join(path,img))
                 
                  img_array=cv2.resize(img_array,(hsize,vsize))
                  img_array=img_array/np.amax(img_array)
                  trainingImageSet[i,:,:,:]=img_array
                  trainingLabelSet[i,:]=class_num
                  i=i+1

    return  trainingImageSet,trainingLabelSet       
# =============================================================================
# Start
# =============================================================================
start = tm.time()
# =============================================================================
# Detailed User Inputs
# =============================================================================
permittedExtension=".jpg"
minimumAccuracy=0.90
initialNeurons=1000
stepsDownRate=1/4
# =============================================================================
# Preprocesamiento de datos
# =============================================================================
categorias=os.listdir(directorioPrincipal)
if "Thumbs.db" in categorias:
   categorias.remove("Thumbs.db")

classnume=len(categorias)
trainingImageSet,trainingLabelSet=create_training_data(directorioPrincipal,categorias,hsize,vsize)
y_train = keras.utils.to_categorical(trainingLabelSet, num_classes=classnume)
# =============================================================================
# Distribucion de datos para entrenamiento
# =============================================================================
X_train, X_test, Y_train, Y_test = train_test_split(
    trainingImageSet, y_train, test_size=0.15, random_state=101)
train_test_split(y_train, shuffle=False)#se evita combinar
# =============================================================================
# Disenio del modelo
# =============================================================================
model = Sequential()
# =============================================================================
# Convolution Layers
# =============================================================================

model.add(Conv2D(32, (3, 3), padding='same',input_shape=(vsize,hsize,3),activation='elu'))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3),activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(40, (3, 3),activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same',activation='elu'))
model.add(Conv2D(64, (3, 3),activation='elu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (2, 2), padding='same',activation='elu'))
model.add(Conv2D(128, (2, 2),activation='elu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(256, (1, 1),activation='elu'))

# =============================================================================
# Fully conected layers
# =============================================================================
model.add(Flatten())

difference=initialNeurons-classnume

for i in range(100):
  model.add(Dense(initialNeurons,activation='elu'))
#  model.add(Dropout(0.3))
  initialNeurons=round(initialNeurons*stepsDownRate)
  if initialNeurons <= classnume:
       break

model.add(Dropout(0.3))
model.add(Dense(classnume, kernel_initializer='normal',activation='softmax'))
# =============================================================================
# Optimizer
# =============================================================================
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00, amsgrad=False)
# =============================================================================
# Compilador
# =============================================================================
model.compile(loss='categorical_crossentropy',#mean_squared_error
              optimizer=adam,
              metrics=['accuracy'])
print(model.summary())
# =============================================================================
# Data augmentation
# =============================================================================
datagen = ImageDataGenerator(
    rotation_range=45,
    zoom_range=-0.2,
    shear_range=0.2)

datagen.fit(X_train)
# =============================================================================
# stopping function for training
# =============================================================================
stopping=EarlyStopping(monitor='acc', min_delta=0.001, patience=10, verbose=0)
# =============================================================================
#Model Training
# =============================================================================
model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),# fits the model on batches with real-time data augmentation:
                    steps_per_epoch=len(X_train)/32, epochs=1,callbacks=[stopping])
# =============================================================================
# Entrenamiento del modelo
# =============================================================================
#model.fit(X_train, Y_train, batch_size=32, epochs=100)
# =============================================================================
# Evaluacion del modelo
# =============================================================================
score = model.evaluate(X_test, Y_test, batch_size=32)
print("Metrics(test loss & Test Accuracy): ")
print(score)
end=tm.time()
horas=((end-start)/60)/60
print('Tiempo de procesamiento [Hr]:')
print(round(horas,5))

# =============================================================================
# Guardar modelo
# =============================================================================
if score[1]>minimumAccuracy:
  scoreString=str(score[1])
  output = tm.strftime("%d-%m-%Y-%H-%M-%S")   
  model.save('C:/Users/H155811/Desktop/SAFRAN'+'modelo_'+(socket.gethostname())+
             '_'+os.path.basename(directorioPrincipal)+'_'+str(hsize)+'x'+str(vsize)+'_Acc_'+scoreString[2:5]+'_T_'+output+'.model')     
  print('Modelo guardado')
else:
    print('El modelo no cumple con los criterios de precision minima requerida')
  