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
import time

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

while True:
    _,imagen = cap.read()#frame sampling
    
    for i in range(0,imageShape[1],horizontalSize):   
        for j in range(0,imageShape[0],verticalSize):    
       
# =============================================================================
# Esto solo es un ejemplo visual
#            tonoGris=255-round(255/random.randrange(1,255))
#            imagen[j:j+verticalSize,i:i+horizontalSize,0]=tonoGris      
# =============================================================================
            
            #LA PORCION DE IMAGEN PARA APLICAR EL MODELO DE DETECCION SERIA:
            imagen2Predict=cv2.resize(imagen[j:j+verticalSize,i:i+horizontalSize,:],(200,200))
            imagen2Predict=imagen2Predict/np.amax(imagen2Predict)
            imageP= np.zeros((1,200,200,3))#revisar reshape
            imageP[0,:,:,:]=imagen2Predict
            predictions = model2.predict(imageP)
                                                  
            confianzamayor=np.argmax(predictions)
            mayor=np.amax(predictions)
#            print(mayor)
            resized_color = cv2.resize(imagen2Predict,(230,230))
            imagen_como_matriz=np.array(resized_color)
            imagen_como_matriz_tamaño_ajustado=imagen_como_matriz.reshape(1,230,230,3)
            y_pred_color=modelo_colores_tornillos.predict_proba(imagen_como_matriz_tamaño_ajustado)
           
            if confianzamayor==1:
                imagen[j:j+verticalSize,i:i+horizontalSize,1]=255 
                cv2.putText(imagen, ('Tornillo '+str(mayor)), (i,j-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
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
            
    cv2.imshow("Neurocode development viewer", imagen)    #ventana mostrando resultados   


  ## Control de acciones a traves del teclado
    k = cv2.waitKey(33)  #si se manda imprimir k en un else aparecera el id de la tecla
    if k==27:    # Esc key to stop
        #falta desconectar camara
        
        break
# ******************************************************************************        
    
cap.release()
cv2.destroyAllWindows()





