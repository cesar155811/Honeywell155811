# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 16:42:54 2020

@author:Neurocode
"""
# =============================================================================
# funcion para descomponer imagenes y aplicar predictor
# =============================================================================
import cv2
import numpy as np
import keras
#import time

model2 = keras.models.load_model('modelo_MX60LT83D9QV2_Labeling_200x200_Acc_983_T_16-04-2020-16-09-33.model') #Poner modelo aqui
modelo_colores_tornillos = keras.models.load_model('SAFRAN_SOLO_COLORES_6.h7')

# # Numero de divisiones deseadas aqui....
divisionesHorizontales=6
divisionesVerticales=5
# ========================
#inicializamos camara
cap = cv2.VideoCapture(0)#camera set up
# =============================================================================
# Meidicion unica para identificar tamanio de la fotografia y calcular sus divisiones
_,imagen = cap.read()#frame sampling
imageShape=imagen.shape #Si se conoce el tamanio de la imagen hacer esta variable constante para ahorrar procesamiento y las dos variables de abajo tmb
horizontalSize=round(imageShape[1]/divisionesHorizontales)  
verticalSize=round(imageShape[0]/divisionesVerticales)
# =============================================================================

imageP= np.zeros((1,200,200,3))#revisar reshape

while True:
#    tic = time.clock()
    _,imagen = cap.read()#frame sampling
    
    for i in range(0,imageShape[1],horizontalSize):   
        for j in range(0,imageShape[0],verticalSize):    
       

            imagen2Predict=cv2.resize(imagen[j:j+verticalSize,i:i+horizontalSize,:],(200,200))
            imagen2Predict=imagen2Predict/np.amax(imagen2Predict)
            imageP[0,:,:,:]=imagen2Predict           
            predictions = model2.predict(imageP)
                                                  
            confianzamayor=np.argmax(predictions)
            mayor=np.amax(predictions)
            resized_color = cv2.resize(imagen2Predict,(230,230))
            imagen_como_matriz=np.array(resized_color)
            imagen_como_matriz_tamaño_ajustado=imagen_como_matriz.reshape(1,230,230,3)
            y_pred_color=modelo_colores_tornillos.predict_proba(imagen_como_matriz_tamaño_ajustado)
           
            if confianzamayor==1 and mayor>0.95:
#                imagen[j:j+verticalSize,i:i+horizontalSize,1]=255    
                cv2.putText(imagen, ('T '+str(round(mayor,3))), (i,j-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                if y_pred_color.item(0)>0.9:
                    cv2.putText(imagen,'ROJO',(i,j-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
                if y_pred_color.item(1)>0.9:
                    cv2.putText(imagen,'ROJO y VERDE',(i,j-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
                if y_pred_color.item(2)>0.9:
                    cv2.putText(imagen,'VERDE',(i,j-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
                if y_pred_color.item(3)>0.9:
                    cv2.putText(imagen,'SIN COLOR',(i,j-25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
 
        # =============================================================================
        # Lineas verdes
    for i in range(horizontalSize,imageShape[1],horizontalSize):      
        imagen = cv2.line(imagen, (i,1), (i,imageShape[0]), (0, 255, 0) , 2)     

            
    for j in range(verticalSize,imageShape[0],verticalSize):   
        imagen = cv2.line(imagen, (1,j), (imageShape[1],j), (0, 255, 0) , 2) 
    # =============================================================================
    
    imagen=cv2.resize(imagen,(1360,880))
         
    cv2.imshow("Neurocode viewer", imagen)    #ventana mostrando resultados   
#    toc = time.clock()
#    print('eltiempo de ciclo es :')
#    print(toc - tic)
    
  ## Control de acciones a traves del teclado
    k = cv2.waitKey(33)  #si se manda imprimir k en un else aparecera el id de la tecla
    if k==27:    # Esc key to stop
        #falta desconectar camara
        
        break
# ******************************************************************************        
    
cap.release()
cv2.destroyAllWindows()





