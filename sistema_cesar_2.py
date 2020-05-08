import cv2
from keras.models import load_model
model_1=load_model('CNN_IMPELLERS_20.h7')
import time

cap=cv2.VideoCapture(0)

while True:
    
    _, frame = cap.read()
    
#    frame1 = cv2.cvtColor(frame)
    
    resize = cv2.resize(frame,(100,100))
    matriz = resize.reshape(1,100,100,3)
    matriz_1 = resize.reshape(1,100,100,3)
    prediccion = model_1.predict(matriz)
    
    if prediccion.item(0)==1:
        print('Numero de parte: 698192-4')
    if prediccion.item(1)==1:
        print('Numero de parte: 3033053-7')
    if prediccion.item(2)==1:
        print('Numero de parte: 3033207-1')
    if prediccion.item(3)==1:
        print('Numero de parte: 3616967-1')
    if prediccion.item(4)==1:
        print('Numero de parte: 3822249-4')
    if prediccion.item(5)==1:
        print('Numero de parte: 3822400-5')
    if prediccion.item(6)==1:
        print('Numero de parte: 3822523-003')
    if prediccion.item(7)==1:
        print('Numero de parte: 70721597-1')
    if prediccion.item(8)==1:
        print('Numero de parte: NO HAY PIEZA')
    if prediccion.item(9)==1:
        print('Numero de parte: LH70027-01')

    #print(prediccion)
    
   # time.sleep(1)
    
    cv2.imshow('frame',frame)
    cv2.imshow('resize',resize)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
