#  Neurocode GUI MAIN CODE
# =============================================================================
# LIBRERIAS
# =============================================================================
import tkinter as tk
import cv2 
import numpy as np
import time as tm
#import winsound
#from matplotlib import pyplot as plt
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import keras
import os
import shutil
import socket
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# =============================================================================
# *****************************************************************************
# =============================================================================

class neurocodeGUI():
    
    def __init__(self):
        
        self.root=tk.Tk()

        # get screen width and height
        self.ws = self.root.winfo_screenwidth() # width of the screen
        self.hs = self.root.winfo_screenheight() # height of the screen
        
        self.w = self.ws*0.59 # width for the Tk root
        self.h = self.hs*0.71 # height for the Tk root        
        
        # calculate x and y coordinates for the Tk root window
        self.x = (self.ws/2) - (self.w/2)
        self.y = (self.hs/2) - (self.h/2)
        
        # set the dimensions of the screen and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (self.w, self.h, self.x, self.y))

        self.root.minsize(250,250)
        self.root.maxsize(int(self.ws*0.59),int(self.hs*0.71))

        self.root.configure(bg='gray13')
        self.root.iconbitmap(default='neuro.ico')#direccion icono
        self.root.title('Neurocode - Artificial Intelligence For Everyone')   
        
## =============================================================================
##   Canvas Background
## =============================================================================
        self.canvasCero = tk.Canvas(self.root, width = self.ws*0.59, height = self.hs*0.71) # Neurocode Logo- windows 2 
        self.canvasCero.place(x=1, y=1)  
        self.back = cv2.imread('NeuroBackBlack.jpg')#direccion logo
        self.back=cv2.resize(self.back,(int(self.ws*0.59),int(self.hs*0.71)))    
        self.backPhoto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.back))
        self.canvasCero.create_image(0, 0, image = self.backPhoto, anchor = tk.NW)              
        
# =============================================================================
#         Canvas
# =============================================================================
        self.canvas = tk.Canvas(self.root, width = 420, height = 100,highlightthickness=0)   # Neurocode Logo- windows 1
        self.canvas.place(x=335, y=15)         
# =============================================================================
#         Ubicacion del logo
# =============================================================================
        self.logo = cv2.imread('neurocodeNegro.jpg')#direccion logo
        self.logo=cv2.resize(self.logo,(425,104))        
        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.logo))
        self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
# =============================================================================
#         Botones iniciales
# =============================================================================

        tk.Button(self.root,height=8, width=20, text='TRAINING', bg='firebrick2',fg='snow',relief='flat',borderwidth=2, command=self.entrenador,font=("Berlin Sans FB", 20)).place(x=100,y=152)        
        tk.Button(self.root,height=8,width=20, text='TEMPLATE\n TESTER', bg='orange',relief='flat',fg='snow',borderwidth=2,command=self.NeuroTemplateTester,font=("Berlin Sans FB", 20)).place(x=412,y=152)        
        tk.Button(self.root,height=8,width=20, text='NEW\n DATA BASE', bg='blue',relief='flat',fg='snow',borderwidth=2, command=self.featureDBC,font=("Berlin Sans FB", 20)).place(x=724,y=415)        
        tk.Button(self.root,height=8,width=20, text='NEURO TESTER', bg='green4',relief='flat',fg='snow',borderwidth=2, command=self.AI_models,font=("Berlin Sans FB", 20)).place(x=724,y=152)        
        tk.Button(self.root,height=8,width=20, text='CLEANER', bg='brown',relief='flat',fg='snow',borderwidth=2, command=self.neuroCleanerButton,font=("Berlin Sans FB", 20)).place(x=100,y=415)        
        tk.Button(self.root,height=8,width=20, text='TEMPLATE\n SAMPLER', bg='purple',relief='flat',fg='snow',borderwidth=2, command=self.templateDBC,font=("Berlin Sans FB", 20)).place(x=412,y=415)
        tk.Button(self.root,text='EXIT',command=self.root.destroy,font=("Berlin Sans FB", 18),bg='gray13',relief='flat',fg='white').place(x=1040,y=700)    
                
        self.root.mainloop()
# =============================================================================
# *****************************************************************************        
# =============================================================================
                    
    def entrenador(self):
        self.root.destroy()
        NeuroTrainer()
      
    def NeuroTemplateTester(self):
        self.root.destroy()     
        AreaTesterClass()
        
# =============================================================================
#     IMAGE PROCESSING
# =============================================================================
             
    def featureDBC(self):
           self.root.destroy()
           featureDataBaseCreator() 
# =============================================================================
#     AI MODELS
# =============================================================================

    def AI_models(self):
       self.root.destroy()        
       NeuroTesterClass()

# =============================================================================
#      CHATBOTS
# =============================================================================
        
    def neuroCleanerButton(self):
       self.root.destroy()        
       Cleaner()
      
# =============================================================================
#      LABELING   
# =============================================================================
        
    def templateDBC(self):
            self.root.destroy()
            TemplateDataBaseCreator() 

# =============================================================================
#     Video Class
# =============================================================================
class MyVideoCapture:
      def __init__(self, video_source):

         self.vid = cv2.VideoCapture(video_source)
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)

         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
      def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             if ret:

                 return (ret, frame)
             else:
                 return (ret, None)
         else:
             return (ret, None)

      def __del__(self):
         if self.vid.isOpened():
             self.vid.release()
             
# =============================================================================
# GUI init               
# =============================================================================
class TemplateDataBaseCreator():
    
    def __init__(self):
        self.root=tk.Tk()
        self.root.configure(bg='gray13')
        self.root.title('Neurocode \ Templates Data Base Generator')
        
        self.ws = self.root.winfo_screenwidth() # width of the screen
        self.hs = self.root.winfo_screenheight()-90 # height of the screen
        self.w=self.ws-270
        self.h=self.hs-150  
    
        # set the dimensions of the screen and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (self.ws, self.hs, 0, 0))         
        self.root.minsize(250,250)  
        self.capturar=0
        self.directorioImageArea=[]
        self.n=4000
        self.tiempo = tm.time()
        self.output = str(int(self.tiempo))

        self.label1=tk.Label(self.root,text="Loading . . .",font=("Arial", 60),bg='gray13',fg='gray80').place(x=740,y=400)

# =============================================================================
#  Parametros iniciales       
# =============================================================================      
        self.start = tm.time()
        
        self.storageFolder = filedialog.askdirectory()
        self.storageFolder=self.storageFolder+'/'  
        self.divisionesHorizontales=5
        self.divisionesVerticales=4
        self.outputViejo=0
        self.counter=0
        
# =============================================================================
# Video Source
# =============================================================================
        self.video_source=1
## =============================================================================
##   Canvas Background
## =============================================================================
        self.canvasCero = tk.Canvas(self.root, width = self.ws, height = 1000) # Neurocode Logo- windows 2 
        self.canvasCero.place(x=0, y=0)  
        self.back = cv2.imread('NeuroBackBlack.jpg')#direccion logo
        self.back=cv2.resize(self.back,(self.ws,1000))    
        self.backPhoto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.back))
        self.canvasCero.create_image(0, 0, image = self.backPhoto, anchor = tk.NW)        

# =============================================================================
#   Canvas Video
# ============================================================================= 
        self.canvasVideo = tk.Canvas(self.root, width = 1280, height = 960, bg="gray13",highlightthickness=1) 
        self.canvasVideo.place(x=340,y=10)
            
# =============================================================================
#   Canvas Logo
# =============================================================================
        self.canvasLogo = tk.Canvas(self.root, width = 213, height = 52,highlightthickness=0) # Neurocode Logo- windows 2 
        self.canvasLogo.place(x=10, y=20)  
        self.logo = cv2.imread('neurocodeNegro.jpg')#direccion logo
        self.logo=cv2.resize(self.logo,(213,52))    
        self.photoLogo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.logo))
        self.canvasLogo.create_image(0, 0, image = self.photoLogo, anchor = tk.NW)            

# =============================================================================
# CLICK EVENT CAMBIO DE TAMANIO CUADRO
# =============================================================================
#        self.canvasVideo.bind("<Button-2>", self.agrandar)
        self.root.bind("<Button-2>", self.agrandar)        
        
# =============================================================================
#  Captura de video         
# =============================================================================
        self.vid = MyVideoCapture(self.video_source)
        
 # =============================================================================
#          Static text
# =============================================================================
        tk.Button(self.root,text='EXIT',command=self.exitAction,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=self.w,y=self.h)
        tk.Button(self.root,text='ACTIVATE',command=self.activate,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=self.w,y=self.h-200)
        tk.Button(self.root,text='FOLDER', bg='gray13',fg='snow',relief='flat',borderwidth=2, command=self.storageFolder_button,font=("Berlin Sans FB", 25)).place(x=self.w,y=self.h-100)                

        ret,self.frame = self.vid.get_frame()  

        imageShape=self.frame.shape 
        self.puntoMedioHorizontal=round(imageShape[1]/2)
        self.puntoMedioVertical=round(imageShape[0]/2)
        self.horizontalSize=round((imageShape[1]/self.divisionesHorizontales)/2)
        self.verticalSize=round((imageShape[0]/self.divisionesVerticales)/2)
        self.reduccion1=0
        self.reduccion2=0
        self.clickCounter=0             
        
        self.delay = 10
        
        self.esqSI=[self.puntoMedioHorizontal-self.horizontalSize+self.reduccion1,self.puntoMedioVertical-self.verticalSize+self.reduccion1]#x,y
        self.esqSD=[self.puntoMedioHorizontal+self.horizontalSize-self.reduccion2,self.puntoMedioVertical-self.verticalSize+self.reduccion1]#x,y
        self.esqII=[self.puntoMedioHorizontal-self.horizontalSize+self.reduccion1,self.puntoMedioVertical+self.verticalSize-self.reduccion1]#x,y
        self.esqID=[self.puntoMedioHorizontal+self.horizontalSize-self.reduccion2,self.puntoMedioVertical+self.verticalSize-self.reduccion1]#x,y                  
        
        
        self.UpdateDataBaseGenerator() 
# =============================================================================
# Loop
# =============================================================================
        self.root.mainloop()
        
# =============================================================================
# folder selection        
# =============================================================================
    def exitAction(self):
        self.root.destroy()
        neurocodeGUI()

    def storageFolder_button(self):
        self.storageFolder = filedialog.askdirectory()
        self.storageFolder=self.storageFolder+'/'  
        self.counter=0
        self.directorioImageArea=[]
        self.tiempo = tm.time()
        self.output = str(int(self.tiempo))
        
        return self.storageFolder

    def activate(self):
        self.tiempo = tm.time()
        self.start2 = tm.time()

        self.output = str(int(self.tiempo))
        
        if self.capturar==1:
            self.capturar=0
        else:    
            self.capturar=1
        return self.capturar

    def agrandar (self,event):
                   
                   self.directorioLargos={0:[0,0],1:[0,30],2:[0,40],3:[30,0],4:[30,30],5:[40,40]}

                   self.reduccion1=self.directorioLargos[self.clickCounter][0]
                   self.reduccion2=self.directorioLargos[self.clickCounter][1]

                   self.clickCounter=self.clickCounter+1
                   
                   if self.clickCounter==5:
                      self.clickCounter=0
                      
# =============================================================================
# Funcion Update      
# =============================================================================
    def UpdateDataBaseGenerator(self): 
        
        ret, self.frame = self.vid.get_frame()    
        self.section=self.frame.copy()
                
        if self.capturar==1:
           self.directorioImageArea.append(self.section)
           self.counter=self.counter+1
        
        if self.capturar==0 and len(self.directorioImageArea)>0:
            for i in range(len(self.directorioImageArea)):
#                print(len(self.directorioImageArea))
                cv2.imwrite(self.storageFolder + self.output + str(i) + '.jpg', self.directorioImageArea[i])  
            self.directorioImageArea=[]                               
                
#        print(len(self.directorioImageArea))        
        
        if len(self.directorioImageArea)==self.n:
            for i in range(self.n):
#                print(len(self.directorioImageArea))
                cv2.imwrite(self.storageFolder + self.output + str(i) + '.jpg', self.directorioImageArea[i])  
            self.directorioImageArea=[]
            self.capturar=0
            self.end2 = tm.time()
            self.lapso2=self.end2 - self.start2
            print("Duracion: "+str(round((self.lapso2/60),4))+" minutos")            
        # =====================================================================
        # Secuencia de Video
        # =====================================================================       
        cv2.rectangle(self.frame,(self.esqSI[0],self.esqSI[1]),(self.esqID[0],self.esqID[1]),(0,255,0),2)              
        cv2.putText(self.frame,str(self.counter),(70,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)                    
        cv2.putText(self.frame,"IMAGES:",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)                    

        self.frame_final=cv2.resize(self.frame,(1280,960))    
        self.frame_final = cv2.cvtColor(self.frame_final, cv2.COLOR_BGR2RGBA)                
        self.photoFinal = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame_final))
        self.canvasVideo.create_image(0,0, image =self.photoFinal, anchor='nw')            
        
        if self.capturar==1:
            self.delay=30
        
        else:
            self.delay=5
        
        self.root.after(self.delay, self.UpdateDataBaseGenerator)                   


             
# =============================================================================
# GUI init               
# =============================================================================
class featureDataBaseCreator():
    
    def __init__(self):
        self.root=tk.Tk()
        self.root.configure(bg='gray13')
        self.root.title('Neurocode \ Feature Data Base Generator')
        
        self.ws = self.root.winfo_screenwidth() # width of the screen
        self.hs = self.root.winfo_screenheight()-90 # height of the screen
        self.w=self.ws-270
        self.h=self.hs-150  
    
        # set the dimensions of the screen and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (self.ws, self.hs, 0, 0))         
        self.root.minsize(250,250)  
        self.capturar=0
        self.directorioImageArea=[]
        self.n=1000
        self.tiempo = tm.time()
        self.output = str(int(self.tiempo))

        self.label1=tk.Label(self.root,text="Loading . . .",font=("Arial", 60),bg='gray13',fg='gray80').place(x=740,y=400)

# =============================================================================
#  Parametros iniciales       
# =============================================================================      
        self.start = tm.time()
        
        self.storageFolder = filedialog.askdirectory()
        self.storageFolder=self.storageFolder+'/'  
        self.divisionesHorizontales=5
        self.divisionesVerticales=4
        self.outputViejo=0
        self.counter=0
# =============================================================================
# Video Source
# =============================================================================
        self.video_source=1
## =============================================================================
##   Canvas Background
## =============================================================================
        self.canvasCero = tk.Canvas(self.root, width = self.ws, height = 1000) # Neurocode Logo- windows 2 
        self.canvasCero.place(x=0, y=0)  
        self.back = cv2.imread('NeuroBackBlack.jpg')#direccion logo
        self.back=cv2.resize(self.back,(self.ws,1000))    
        self.backPhoto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.back))
        self.canvasCero.create_image(0, 0, image = self.backPhoto, anchor = tk.NW)        

# =============================================================================
#   Canvas Video
# ============================================================================= 
        self.canvasVideo = tk.Canvas(self.root, width = 1280, height = 960, bg="gray13",highlightthickness=1) 
        self.canvasVideo.place(x=340,y=10)
            
# =============================================================================
#   Canvas Logo
# =============================================================================
        self.canvasLogo = tk.Canvas(self.root, width = 213, height = 52,highlightthickness=0) # Neurocode Logo- windows 2 
        self.canvasLogo.place(x=10, y=20)  
        self.logo = cv2.imread('neurocodeNegro.jpg')#direccion logo
        self.logo=cv2.resize(self.logo,(213,52))    
        self.photoLogo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.logo))
        self.canvasLogo.create_image(0, 0, image = self.photoLogo, anchor = tk.NW)            

# =============================================================================
# CLICK EVENT CAMBIO DE TAMANIO CUADRO
# =============================================================================
#        self.canvasVideo.bind("<Button-2>", self.agrandar)
        self.root.bind("<Button-2>", self.agrandar)        
        
# =============================================================================
#  Captura de video         
# =============================================================================
        self.vid = MyVideoCapture(self.video_source)
        
 # =============================================================================
#          Static text
# =============================================================================
        tk.Button(self.root,text='EXIT',command=self.exitAction,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=self.w,y=self.h)
        tk.Button(self.root,text='ACTIVATE',command=self.activate,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=self.w,y=self.h-200)
        tk.Button(self.root,text='FOLDER', bg='gray13',fg='snow',relief='flat',borderwidth=2, command=self.storageFolder_button,font=("Berlin Sans FB", 25)).place(x=self.w,y=self.h-100)                

        ret,self.frame = self.vid.get_frame()  

        imageShape=self.frame.shape 
        self.puntoMedioHorizontal=round(imageShape[1]/2)
        self.puntoMedioVertical=round(imageShape[0]/2)
        self.horizontalSize=round((imageShape[1]/self.divisionesHorizontales)/2)
        self.verticalSize=round((imageShape[0]/self.divisionesVerticales)/2)
        self.reduccion1=0
        self.reduccion2=0
        self.clickCounter=0             
        
        self.delay = 10
        self.UpdateDataBaseGenerator() 
# =============================================================================
# Loop
# =============================================================================
        self.root.mainloop()
        
# =============================================================================
# folder selection        
# =============================================================================
    def exitAction(self):
        self.root.destroy()
        neurocodeGUI()

    def storageFolder_button(self):
        self.storageFolder = filedialog.askdirectory()
        self.storageFolder=self.storageFolder+'/'  
        self.counter=0
        self.directorioImageArea=[]
        self.tiempo = tm.time()
        self.output = str(int(self.tiempo))
        
        return self.storageFolder

    def activate(self):
        self.tiempo = tm.time()
        self.start2 = tm.time()

        self.output = str(int(self.tiempo))
        
        if self.capturar==1:
            self.capturar=0
        else:    
            self.capturar=1
        return self.capturar

    def agrandar (self,event):
                   
                   self.directorioLargos={0:[0,0],1:[0,30],2:[0,40],3:[30,0],4:[30,30],5:[40,40],6:[50,50],7:[60,60]}

                   self.reduccion1=self.directorioLargos[self.clickCounter][0]
                   self.reduccion2=self.directorioLargos[self.clickCounter][1]

                   self.clickCounter=self.clickCounter+1
                   
                   if self.clickCounter==7:
                      self.clickCounter=0
                      
# =============================================================================
# Funcion Update      
# =============================================================================
    def UpdateDataBaseGenerator(self): 

        self.esqSI=[self.puntoMedioHorizontal-self.horizontalSize+self.reduccion1,self.puntoMedioVertical-self.verticalSize+self.reduccion2]#x,y
        self.esqSD=[self.puntoMedioHorizontal+self.horizontalSize-self.reduccion1,self.puntoMedioVertical-self.verticalSize+self.reduccion2]#x,y
        self.esqII=[self.puntoMedioHorizontal-self.horizontalSize+self.reduccion1,self.puntoMedioVertical+self.verticalSize-self.reduccion2]#x,y
        self.esqID=[self.puntoMedioHorizontal+self.horizontalSize-self.reduccion1,self.puntoMedioVertical+self.verticalSize-self.reduccion2]#x,y  
        
        ret, self.frame = self.vid.get_frame()    
        self.framecopy=self.frame.copy()
        self.section=self.framecopy[self.esqSI[1]:self.esqID[1],self.esqSI[0]:self.esqID[0],:]
                
        if self.capturar==1:
           self.directorioImageArea.append(self.section)
           self.counter=self.counter+1
        
        if self.capturar==0 and len(self.directorioImageArea)>0:
            for i in range(len(self.directorioImageArea)):
#                print(len(self.directorioImageArea))
                cv2.imwrite(self.storageFolder + self.output + str(i) + '.jpg', self.directorioImageArea[i])  
            self.directorioImageArea=[]                               
                
#        print(len(self.directorioImageArea))        
        
        if len(self.directorioImageArea)==self.n:
            for i in range(self.n):
#                print(len(self.directorioImageArea))
                cv2.imwrite(self.storageFolder + self.output + str(i) + '.jpg', self.directorioImageArea[i])  
            self.directorioImageArea=[]
            self.capturar=0
            self.end2 = tm.time()
            self.lapso2=self.end2 - self.start2
            print("Duracion: "+str(round((self.lapso2/60),4))+" minutos")            
        # =====================================================================
        # Secuencia de Video
        # =====================================================================       
        cv2.rectangle(self.frame,(self.esqSI[0],self.esqSI[1]),(self.esqID[0],self.esqID[1]),(0,255,0),2)              
        cv2.putText(self.frame,str(self.counter),(70,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)                    
        cv2.putText(self.frame,"IMAGES:",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)                    

        self.frame_final=cv2.resize(self.frame,(1280,960))    
        self.frame_final = cv2.cvtColor(self.frame_final, cv2.COLOR_BGR2RGBA)                
        self.photoFinal = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame_final))
        self.canvasVideo.create_image(0,0, image =self.photoFinal, anchor='nw')            
        
        if self.capturar==1:
            self.delay=30
        
        else:
            self.delay=5
        
        self.root.after(self.delay, self.UpdateDataBaseGenerator)                   

# =============================================================================
# GUI init               
# =============================================================================
class NeuroTesterClass():
    
    def __init__(self):
        self.root=tk.Tk()
        self.root.configure(bg='gray13')
        self.root.title('Neurocode \ NeuroTester')
        
        self.ws = self.root.winfo_screenwidth() # width of the screen
        self.hs = self.root.winfo_screenheight()-90 # height of the screen
        self.w=self.ws
        self.h=self.hs 
    
        # set the dimensions of the screen and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (self.ws, self.hs, 0, 0))         
        self.root.minsize(250,250)  
   
        self.label1=tk.Label(self.root,text="Loading . . .",font=("Arial", 60),bg='gray13',fg='gray80').place(x=740,y=400)
# =============================================================================
#   Imagen match
# =============================================================================
        self.directorioImageArea=[]
        self.directorioImageArea.append (cv2.imread('EPX1'))
        self.directorioImageArea.append (cv2.imread('EPX2'))
        self.directorioImageArea.append (cv2.imread('EPX2-2'))
        self.directorioImageArea.append (cv2.imread('EPX3'))
        self.directorioImageArea.append (cv2.imread('EPX3-2'))
        self.directorioImageArea.append (cv2.imread('EPX1'))
# =============================================================================
#  Parametros iniciales       
# =============================================================================      
        self.color1=(0,255,255)
        self.color2=(0,255,255)
        self.color3=(0,255,255)
        self.color4=(0,255,255)
        self.color5=(0,255,255)

        self.start = tm.time()
        self.divisionesHorizontales=5
        self.divisionesVerticales=4
        
        
        self.modelo1 = keras.models.load_model('modelo_MX60LT83D9QV2_Colores de Tornillo_LXL100x100_Loss_001_Acc_0_T_28-06-2020-18-25-57.model') #Poner modelo aqui
        self.directorio1={0:"FALTA MARCA",1:"TORNILLO CON MARCA",2:"",3:"",4:"",5:"",6:"",7:"",8:"",9:"",10:"",11:"",12:""}

        self.modelo2 = keras.models.load_model('modelo_MX60LT83D9QV2_Componentes Clean2_LXL100x100_Loss_025_Acc_996_T_26-06-2020-17-27-30.model') #Poner modelo aqui
        self.directorio2={0:"CANDADO",1:"SEARCHING...",2:"EPX4_1",3:"EPX4_2",4:"RELOJ EPX1",5:"RELOJ EPX2",6:"RELOJ EPX2-2",7:"RELOJ EPX3",8:"RELOJ EPX3-2",9:"RELOJ NO EXISTENTE",10:"SIN TORNILLO",11:"TORNILLO CON ARANDELAS",12:"TORNILLO SIN ARANDELA"}

        self.modelo3 = keras.models.load_model('modelo_MX60LT83D9QV2_EPX_CESAR_LXL100x100_Loss_000_Acc_0_T_29-06-2020-16-55-35.model')
        self.directorio3={0:"EPX",1:"EPX2",2:"EPX2_2",3:"EPX3",4:"EPX3_3",5:"EPX3_3",6:"EPX4_1",7:"EPX4_2",8:"",9:"",10:"",11:"",12:""}

#        self.modelo2 = keras.models.load_model('modelo_MX60LT83D9QV2_SoloTornillos_100x100_Loss_013_Acc_995_T_19-06-2020-17-26-51.model') #Poner modelo aqui
#        self.directorio2={0:"Searching...",1:"Sin_tornillo",2:"Tornillos con arandelas",3:"Tornillo sin arandelas",4:"",5:"",6:"",7:"",8:"",9:"",10:"",11:"",12:""}

#        self.modelo1 = keras.models.load_model('modelo_MX60LT83D9QV2_Colores de Tornillo_100x100_Loss_000_Acc_0_T_19-06-2020-16-31-14.model') #Poner modelo aqui
#        self.directorio1={0:"FALTA MARCA",1:"TORNILLO CON MARCA"}

        self.tamañodeLado1=100
        self.valorMediana=30
        
        self.medianValueArray=[]
        self.counter=0
        self.mediana=0
        self.holder=0
        self.texto1=" "        
        self.image1= np.zeros((1,self.tamañodeLado1,self.tamañodeLado1,3))#revisar reshape
        self.modeloElegido=1
# =============================================================================
# Video Source
# =============================================================================
        self.video_source=1
## =============================================================================
##   Canvas Background
## =============================================================================
        self.canvasCero = tk.Canvas(self.root, width = self.ws, height = 1000) # Neurocode Logo- windows 2 
        self.canvasCero.place(x=0, y=0)  
        self.back = cv2.imread('NeuroBackBlack.jpg')#direccion logo
        self.back=cv2.resize(self.back,(self.ws,1000))    
        self.backPhoto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.back))
        self.canvasCero.create_image(0, 0, image = self.backPhoto, anchor = tk.NW)        

# =============================================================================
#   Canvas Video
# ============================================================================= 
        self.canvasVideo = tk.Canvas(self.root, width = 1280, height = 960, bg="gray13",highlightthickness=3) 
        self.canvasVideo.place(x=370,y=10)
            
# =============================================================================
#   Canvas Logo
# =============================================================================
        self.canvasLogo = tk.Canvas(self.root, width = 213, height = 52,highlightthickness=0) # Neurocode Logo- windows 2 
        self.canvasLogo.place(x=10, y=20)  
        self.logo = cv2.imread('neurocodeNegro.jpg')#direccion logo
        self.logo=cv2.resize(self.logo,(213,52))    
        self.photoLogo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.logo))
        self.canvasLogo.create_image(0, 0, image = self.photoLogo, anchor = tk.NW)            
# =============================================================================
##   Canvas ImagenGuia
## =============================================================================
#        self.canvasGuia1 = tk.Canvas(self.root, width = 640, height = 480, highlightthickness=1) #Guide image
#        self.canvasGuia1.place(x=5, y=250)        
        self.imagenGuia1 = cv2.imread('EPX1 AREA.jpg')     
        self.imagenGuia1=cv2.resize(self.imagenGuia1,(426,320))    
#        self.imagenGuia1 = cv2.cvtColor(self.imagenGuia1, cv2.COLOR_BGR2RGBA)                
#        self.photoGuia1 = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.imagenGuia1))
#        self.canvasGuia1.create_image(0, 0, image = self.photoGuia1, anchor = tk.NW)  
# =============================================================================
#  Captura de video         
# =============================================================================
        self.vid = MyVideoCapture(self.video_source)
        
# =============================================================================
#          Static text
# =============================================================================
        tk.Button(self.root,text='EXIT',command=self.root.destroy,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=self.w-155,y=self.h-100)
        tk.Button(self.root,text='CHANGE MODEL',command=self.cambiarModelo,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=5,y=self.h-200)
#        tk.Button(self.root,text='START',font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=5,y=self.h-400)

        ret,self.frame = self.vid.get_frame()  

        self.imageShape=self.frame.shape 
        self.puntoMedioHorizontal=round(self.imageShape[1]/2)
        self.puntoMedioVertical=round(self.imageShape[0]/2)
        self.horizontalSize=round((self.imageShape[1]/self.divisionesHorizontales)/2)
        self.verticalSize=round((self.imageShape[0]/self.divisionesVerticales)/2)
        self.reduccion=0
                
        
        self.delay = 10
        self.UpdateNeuroTester() 
# =============================================================================
# Loop
# =============================================================================
        self.root.mainloop()


    def cambiarModelo(self):
 
        self.color1=(0,255,255)
        self.color2=(0,255,255)
        self.color3=(0,255,255)
        self.color4=(0,255,255)
        self.color5=(0,255,255)
        
        if self.modeloElegido==1:
          self.modeloElegido=2
        else:
          self.modeloElegido=1
         
# =============================================================================
# Funcion Update      
# =============================================================================
    def UpdateNeuroTester(self): 
        
        ret, self.frame = self.vid.get_frame()        
# =============================================================================
# Deteccion de area
# =============================================================================
#        self.imagen2Predict=cv2.resize(self.frame,(self.tamañodeLado1,self.tamañodeLado1))
#        self.imagen2Predict=self.imagen2Predict/np.amax(self.imagen2Predict)
#        self.image1[0,:,:,:]=self.imagen2Predict
#        self.predictions = self.modelo3.predict(self.image1)
#        self.confianzamayor=np.argmax(self.predictions)
#        self.mayor=np.amax(self.predictions)    
#        
#        if self.mayor>0.95:
#            self.zonaElegida=self.confianzamayor
#             
#            if self.zonaElegida==0:
#                res = cv2.matchTemplate(self.frame,self.directorioImageArea[self.zonaElegida],cv2.TM_CCOEFF_NORMED)   
#                print(res)
#                if res>0.5:
#                      print("el res es: "+ str(res)) 
#                      self.color1=(255,255,0)
#    
#            if self.zonaElegida==1:
#                #else if match
#                self.color2=(0,255,0)            
#    
#            if self.zonaElegida==2:
#                #else if match
#                self.color3=(0,255,0)        
#    
#            if self.zonaElegida==3:
#                #else if match
#                self.color4=(0,255,0)               
#    
#            if self.zonaElegida==4:
#                #else if match
#                self.color5=(0,255,0)               
            
# =============================================================================
# #########################################################################33   
# =============================================================================
      
        self.esqSI=[self.puntoMedioHorizontal-self.horizontalSize+self.reduccion,self.puntoMedioVertical-self.verticalSize+self.reduccion]#x,y
        self.esqSD=[self.puntoMedioHorizontal+self.horizontalSize-self.reduccion,self.puntoMedioVertical-self.verticalSize+self.reduccion]#x,y
        self.esqII=[self.puntoMedioHorizontal-self.horizontalSize+self.reduccion,self.puntoMedioVertical+self.verticalSize-self.reduccion]#x,y
        self.esqID=[self.puntoMedioHorizontal+self.horizontalSize-self.reduccion,self.puntoMedioVertical+self.verticalSize-self.reduccion]#x,y        
        
        self.section=self.frame[self.esqSI[1]:self.esqID[1],self.esqSI[0]:self.esqID[0],:]

        self.imagen2Predict=cv2.resize(self.section,(self.tamañodeLado1,self.tamañodeLado1))
        self.imagen2Predict=self.imagen2Predict/np.amax(self.imagen2Predict)
        self.image1[0,:,:,:]=self.imagen2Predict
       
        if self.modeloElegido==1:
          self.predictions = self.modelo1.predict(self.image1)

        if self.modeloElegido==2:
          self.predictions = self.modelo2.predict(self.image1)        
        
        self.confianzamayor=np.argmax(self.predictions)
        self.mayor=np.amax(self.predictions)
    
        if self.mayor>0.95 and self.holder==self.confianzamayor:
                 self.counter=self.counter+1       
                 self.medianValueArray.append(self.confianzamayor)
                 self.frame[self.esqSI[1]:self.esqID[1],self.esqSI[0]:self.esqID[0],1:2]=255     
    
                 if self.counter==self.valorMediana:  
                     self.medianValueArray.sort()
                     self.mediana=self.medianValueArray[round(len(self.medianValueArray)/2)]
    #                 texto=str(directorio1[mediana])+" valor directorio: "+str(mediana)
                     self.medianValueArray=[]
                     self.counter=0

                 if self.modeloElegido==1:
                   self.texto1=self.directorio1[self.confianzamayor]
        
                 if self.modeloElegido==2:
                   self.texto1=self.directorio2[self.confianzamayor]
                 
#                 winsound.Beep(400, 180)

    
        if self.mayor>0.95 and self.holder!=self.confianzamayor:
    #           print("conteos positivos "+str(counter))
    #           texto1=" "
               self.holder=self.confianzamayor
            
    #    cv2.putText(imagen,texto1,(esqSD[0]+5,esqSD[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(self.frame,self.texto1,(self.puntoMedioHorizontal-300,self.puntoMedioVertical+230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

                
        # =====================================================================
        # Secuencia de Video
        # =====================================================================       
        cv2.rectangle(self.frame,(self.esqSI[0],self.esqSI[1]),(self.esqID[0],self.esqID[1]),(0,255,0),2)     
        self.texto2="LIVE DETECTION"+' '+str(round(self.mayor,5))
        cv2.putText(self.frame,self.texto2,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)                    

        self.frame_final=cv2.resize(self.frame,(1280,960))    
        
#rectangulos indicadores amarillos
        ##############################################################
        cv2.rectangle(self.imagenGuia1,(11,37),(61,87),self.color1,4)     
        cv2.rectangle(self.imagenGuia1,(375,37),(425,87),self.color2,4)     
        cv2.rectangle(self.imagenGuia1,(0,197),(49,247),self.color3,4)     
        cv2.rectangle(self.imagenGuia1,(188,176),(238,226),self.color4,4)     
        cv2.rectangle(self.imagenGuia1,(384,202),(426,252),self.color5,4)     
        ################################################################
        self.frame_final[0:320,1280-426:1280,:]=self.imagenGuia1
        punto1x=1280-426
        cv2.rectangle(self.frame_final,(punto1x,0),(1280,320),(255,255,255),2)     
        

        self.frame_final = cv2.cvtColor(self.frame_final, cv2.COLOR_BGR2RGBA)                
        self.photoFinal = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame_final))
        self.canvasVideo.create_image(0,0, image =self.photoFinal, anchor='nw')            
        
        
        self.root.after(self.delay, self.UpdateNeuroTester)                                                 

# =============================================================================
class Cleaner():
    
    def __init__(self):
        self.root=tk.Tk()
        self.root.configure(bg='gray13')
        self.root.title('Neurocode \ Data Base Cleaner')
        #extraccion de tamanio de la pantalla
        self.ws = self.root.winfo_screenwidth() # width of the screen
        self.hs = self.root.winfo_screenheight() # height of the screen
        # set the dimensions of the screen and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (self.ws, self.hs, 0, 0))         
        self.root.minsize(int(self.ws/8),int(self.hs/8))       
        
## =============================================================================
##   Canvas Background
## =============================================================================
        self.canvasCero = tk.Canvas(self.root, width = self.ws*0.99, height = self.hs*0.99) # Neurocode Logo- windows 2 
        self.canvasCero.place(x=1, y=1)  
        self.back = cv2.imread('NeuroBackBlack.jpg')#direccion logo
        self.back=cv2.resize(self.back,(int(self.ws*0.99),int(self.hs*0.99)))    
        self.backPhoto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.back))
        self.canvasCero.create_image(0, 0, image = self.backPhoto, anchor = tk.NW)                    
# =============================================================================
#   Canvas Logo
# =============================================================================
        self.canvasLogo = tk.Canvas(self.root, width = int(self.ws*0.15), height =int(self.hs*0.07),highlightthickness=0) # Neurocode Logo- windows 2 
        self.canvasLogo.place(x=int(self.ws*0.05), y=int(self.hs*0.05))  
        self.logo = cv2.imread('neurocodeNegro.jpg')#direccion logo
        self.logo=cv2.resize(self.logo,(int(self.ws*0.15),int(self.hs*0.07)))    
        self.photoLogo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.logo))
        self.canvasLogo.create_image(0, 0, image = self.photoLogo, anchor = tk.NW)     
# =============================================================================
   #  Botones y texto
# =============================================================================
        self.button1 = tk.Button(self.root, height=int(self.hs*0.005), width=int(self.ws*0.01),text="CLEAN DATA", bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.CleanImagesInFolders,font=("Berlin Sans FB", 30),state=tk.DISABLED)
        self.button1.place(x=int(self.ws*0.56),y=int(self.hs*0.38)) 

        self.button2 = tk.Button(self.root, height=int(self.hs*0.005), width=int(self.ws*0.01),text="FINISH\n REVIEW", bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.CleanFinish,font=("Berlin Sans FB", 30),state=tk.DISABLED)
        self.button2.place(x=int(self.ws*0.38),y=int(self.hs*0.66)) 
        self.button0=tk.Button(self.root,height=int(self.hs*0.005), width=int(self.ws*0.01), text='SELECT MODEL', bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.modelSelect,font=("Berlin Sans FB", 30),state=tk.NORMAL)
        self.button0.place(x=int(self.ws*0.20),y=int(self.hs*0.38))        

        tk.Button(self.root,text='EXIT',command=self.exitAction,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=int(self.ws*0.885),y=int(self.hs*0.70))

        # =============================================================================
        #creacion de folder para cuarentena
        self.storageFolder="C:/TEMP/Carpeta de fotos malas"        
        self.isdir = os.path.isdir(self.storageFolder)         
        if self.isdir ==False:
           os.mkdir(self.storageFolder)  
           
        # =============================================================================
        #   Parametros iniciales
        # =============================================================================
        self.indicadorAcumulado=0
        self.fotosEvaluadas=0
        self.counter1=0
        self.counter2=0
        self.directorioKey=dict()
# =============================================================================
#         Main loop
# =============================================================================
        self.root.mainloop()

    def exitAction(self):
        self.root.destroy()
        neurocodeGUI()
        
    def modelSelect(self):
        self.filename = filedialog.askopenfilename()
#        self.modelo1text=os.path.basename(self.filename)
        self.modelo1=keras.models.load_model(self.filename) 
        # =============================================================================
        self.placeString=self.filename.find("LXL")
        self.placeStringEnd=self.filename.find("x")
        self.tamanioLadoModelo=int(self.filename[self.placeString+3:self.placeStringEnd])     
        self.imageP= np.zeros((1,self.tamanioLadoModelo,self.tamanioLadoModelo,3)) 
        self.button1['state'] = tk.NORMAL     
        self.button0['state'] = tk.DISABLED 

    def CleanFinish(self):
        self.carpetas=os.listdir(self.mainParentFolder)
        
        for folder in self.carpetas:
            self.pathCarpetaMalas= self.storageFolder+'/'+folder+"_Revision"
            self.pathCarpetaBuena=self.mainParentFolder+'/'+folder
            
            self.Listafotos=os.listdir(os.path.join(self.pathCarpetaMalas))
            
            for fotos in self.Listafotos:
                self.imagenPath=self.pathCarpetaMalas+"/"+fotos
                shutil.copy(self.imagenPath, self.pathCarpetaBuena)
                os.remove(self.imagenPath)
        
        self.carpetas=os.listdir(self.storageFolder)
        
        for folder in self.carpetas:
            self.pathCarpetaMalas= self.storageFolder+'/'+folder    
            os.rmdir(self.pathCarpetaMalas)      
            
        self.button2['state'] = tk.DISABLED 
        self.button0['state'] = tk.NORMAL 

     
    def CleanImagesInFolders(self):  
        self.filename = filedialog.askopenfilename()
        self.directorio=os.path.dirname(self.filename)
        self.mainParentFolder=os.path.abspath(os.path.join(self.directorio, '..'))
        print("\n")  
        print("El folder principal donde estan las carpetas evaluadas es: "+self.mainParentFolder)
        print("\n")                
        self.foldersConCaracteristicas=os.listdir(self.mainParentFolder)
        print("Las caracteristicas a evaluar son: ")
        print(self.foldersConCaracteristicas)
        print("\n")   
                
        self.counterFolder=0
        
        for i in range(len(self.foldersConCaracteristicas)):
            self.directorioKey[i] = self.foldersConCaracteristicas[i]
             
        print(self.directorioKey)
        print("\n")   
        
        for carpeta in self.foldersConCaracteristicas:
            self.incorrectos=0
            print("El folder evaluado en esta iteracion es: "+os.path.join(self.mainParentFolder,carpeta))
            print("\n")   
            self.imagenes=os.listdir(os.path.join(self.mainParentFolder,carpeta)) 
            self.conteoDImagen=len(self.imagenes)
            print("Imagenes encontradas en directorio "+carpeta+": "+str(len(self.imagenes)))     
            self.folderEnCuarentena=self.storageFolder+"/"+carpeta+"_Revision"
            self.isdir = os.path.isdir(self.folderEnCuarentena) 
            if self.isdir ==False:
               os.mkdir(self.folderEnCuarentena)                         

            self.indiceCategory=self.foldersConCaracteristicas.index(carpeta)    
    
            self.remover=0    
            if "Thumbs.db" in self.imagenes:
               print('\n Se encontro archivo Thumbs.db y fue removido')
               self.remover=1
    
            self.valor=len(self.imagenes)-self.remover 
            for i in range(self.valor):
              self.strings=os.path.join(self.mainParentFolder,carpeta,self.imagenes[i])
              self.image=cv2.imread(self.strings)
              self.imagen2Predict=cv2.resize(self.image,(self.tamanioLadoModelo,self.tamanioLadoModelo))
              self.imagen2Predict=self.imagen2Predict/np.amax(self.imagen2Predict)
              self.imageP[0,:,:,:]=self.imagen2Predict
              self.predictions = self.modelo1.predict(self.imageP)
#              print(self.predictions)
              self.confianzamayor=np.argmax(self.predictions)
              
              self.mayor=np.amax(self.predictions)
              if self.confianzamayor == self.indiceCategory:
                   self.counter1=self.counter1+1      
                   
              if self.confianzamayor != self.indiceCategory:
#                   plt.imshow(self.imagen2Predict)
#                   plt.show()   
                   print("\n")
                   print(self.strings)
                   print("valor mayor: "+str(self.mayor)+" indice: "+str(self.confianzamayor))
                   print("el valor de confianza dice que es: "+str(self.directorioKey[self.confianzamayor])+" y deberia ser: "+str(self.directorioKey[self.indiceCategory]))
                   shutil.copy(self.strings, self.folderEnCuarentena)
                   print("Archivo llevado a carpeta de inspeccion")
                   os.remove(self.strings)     
                   self.incorrectos=self.incorrectos+1
                   self.counter2=self.counter2+1
                   
            print("Imagenes incorrectas en carpeta: "+str(self.incorrectos))   
            print("\n")
            self.fotosEvaluadas=self.fotosEvaluadas+self.conteoDImagen
            
        print('imagenes identificadas correctamente:'+str(self.counter1))       
        print('imagenes identificadas incorrectamente:'+str(self.counter2))       
        self.indicadorAcumulado=self.counter1+self.counter2   
        print("Total de imagenes evaluadas: "+str(self.fotosEvaluadas))
        self.rate=(self.counter2/self.indicadorAcumulado)*100
        print("Indicador performance: "+str(round(self.rate,5)))
        os.startfile(self.storageFolder)                
        self.button1['state'] = tk.DISABLED
        self.button2['state'] = tk.NORMAL 
        self.button0['state'] = tk.DISABLED 
        
# =============================================================================
# GUI init               
# =============================================================================
class AreaTesterClass():
    
    def __init__(self):
        self.root=tk.Tk()
        self.root.configure(bg='gray13')
        self.root.title('Neurocode \ NeuroTester')
        
        self.ws = self.root.winfo_screenwidth() # width of the screen
        self.hs = self.root.winfo_screenheight()-90 # height of the screen
        self.w=self.ws
        self.h=self.hs 
    
        # set the dimensions of the screen and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (self.ws, self.hs, 0, 0))         
        self.root.minsize(250,250)  
   
        self.label1=tk.Label(self.root,text="Loading . . .",font=("Arial", 60),bg='gray13',fg='gray80').place(x=740,y=400)
## =============================================================================
##   Imagen match
## =============================================================================
#        self.directorioImageArea=[]
#        self.directorioImageArea.append (cv2.imread('EPX1'))
#        self.directorioImageArea.append (cv2.imread('EPX2'))
#        self.directorioImageArea.append (cv2.imread('EPX2-2'))
#        self.directorioImageArea.append (cv2.imread('EPX3'))
#        self.directorioImageArea.append (cv2.imread('EPX3-2'))
#        self.directorioImageArea.append (cv2.imread('EPX1'))
# =============================================================================
#  Parametros iniciales       
# =============================================================================      

        self.start = tm.time()
        self.divisionesHorizontales=5
        self.divisionesVerticales=4
          
#        self.modelo1 = keras.models.load_model('modelo_MX60LT83D9QV2_Colores de Tornillo_100x100_Loss_001_Acc_0_T_28-06-2020-18-25-57.model') #Poner modelo aqui
#        self.directorio1={0:"FALTA MARCA",1:"TORNILLO CON MARCA",2:"",3:"",4:"",5:"",6:"",7:"",8:"",9:"",10:"",11:"",12:""}
#
        self.modelo1 = keras.models.load_model('modelo_MX60LT83D9QV2_cubiculo_LXL100x100_Loss_005_Acc_0_T_06-07-2020-14-25-32.model') #Poner modelo aqui
        self.directorio1={0:"Cubiculo general",1:"compu",2:"conector",3:"foto",4:"mochila",5:"plato",6:"",7:"",8:"",9:"",10:"",11:"",12:""}


        self.tamañodeLado1=100
        self.image1= np.zeros((1,self.tamañodeLado1,self.tamañodeLado1,3))#revisar reshape

        self.valorMediana=30
        
        self.medianValueArray=[]
        self.counter=0
        self.mediana=0
        self.holder=0
        self.texto1=" "        
        self.modeloElegido=1
# =============================================================================
# Video Source
# =============================================================================
        self.video_source=1
## =============================================================================
##   Canvas Background
## =============================================================================
        self.canvasCero = tk.Canvas(self.root, width = self.ws, height = 1000) # Neurocode Logo- windows 2 
        self.canvasCero.place(x=0, y=0)  
        self.back = cv2.imread('NeuroBackBlack.jpg')#direccion logo
        self.back=cv2.resize(self.back,(self.ws,1000))    
        self.backPhoto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.back))
        self.canvasCero.create_image(0, 0, image = self.backPhoto, anchor = tk.NW)        

# =============================================================================
#   Canvas Video
# ============================================================================= 
        self.canvasVideo = tk.Canvas(self.root, width = 1280, height = 960, bg="gray13",highlightthickness=3) 
        self.canvasVideo.place(x=370,y=10)
            
# =============================================================================
#   Canvas Logo
# =============================================================================
        self.canvasLogo = tk.Canvas(self.root, width = 213, height = 52,highlightthickness=0) # Neurocode Logo- windows 2 
        self.canvasLogo.place(x=10, y=20)  
        self.logo = cv2.imread('neurocodeNegro.jpg')#direccion logo
        self.logo=cv2.resize(self.logo,(213,52))    
        self.photoLogo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.logo))
        self.canvasLogo.create_image(0, 0, image = self.photoLogo, anchor = tk.NW)            

# =============================================================================
#  Captura de video         
# =============================================================================
#        self.vid = MyVideoCapture(self.video_source)
        self.videoActivado=cv2.VideoCapture(self.video_source)

# =============================================================================
#          Static text
# =============================================================================
        tk.Button(self.root,text='EXIT',command=self.root.destroy,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=self.w-155,y=self.h-100)

        ret,self.frame = self.videoActivado.read() 

        self.imageShape=self.frame.shape 
        self.puntoMedioHorizontal=round(self.imageShape[1]/2)
        self.puntoMedioVertical=round(self.imageShape[0]/2)
        self.horizontalSize=round((self.imageShape[1]/self.divisionesHorizontales)/2)
        self.verticalSize=round((self.imageShape[0]/self.divisionesVerticales)/2)
        self.reduccion=0
        
        self.delay = 10
        self.UpdateNeuroTester() 
                
# =============================================================================
# Loop
# =============================================================================
        self.root.mainloop()
         
# =============================================================================
# Funcion Update      
# =============================================================================
    def UpdateNeuroTester(self): 
        
        
        _,self.frame = self.videoActivado.read()        
# =============================================================================
      
        self.section=self.frame.copy()

        self.imagen2Predict=cv2.resize(self.section,(self.tamañodeLado1,self.tamañodeLado1))
        self.imagen2Predict=self.imagen2Predict/np.amax(self.imagen2Predict)
        self.image1[0,:,:,:]=self.imagen2Predict
       
        if self.modeloElegido==1:
          self.predictions = self.modelo1.predict(self.image1)

        if self.modeloElegido==2:
          self.predictions = self.modelo2.predict(self.image1)        
        
        self.confianzamayor=np.argmax(self.predictions)
        self.mayor=np.amax(self.predictions)
    
        if self.mayor>0.95 and self.holder==self.confianzamayor:
                 self.counter=self.counter+1       
                 self.medianValueArray.append(self.confianzamayor)
    
                 if self.counter==self.valorMediana:  
                     self.medianValueArray.sort()
                     self.mediana=self.medianValueArray[round(len(self.medianValueArray)/2)]
    #                 texto=str(directorio1[mediana])+" valor directorio: "+str(mediana)
                     self.medianValueArray=[]
                     self.counter=0

                 if self.modeloElegido==1:
                   self.texto1=self.directorio1[self.confianzamayor]
        
                 if self.modeloElegido==2:
                   self.texto1=self.directorio2[self.confianzamayor]
                 
#                 winsound.Beep(400, 180)
   
        if self.mayor>0.95 and self.holder!=self.confianzamayor:
    #           print("conteos positivos "+str(counter))
    #           texto1=" "
               self.holder=self.confianzamayor
            
    #    cv2.putText(imagen,texto1,(esqSD[0]+5,esqSD[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(self.frame,self.texto1,(self.puntoMedioHorizontal-300,self.puntoMedioVertical+230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
         
        # =====================================================================
        # Secuencia de Video
        # =====================================================================       
        self.texto2="LIVE DETECTION"+' '+str(round(self.mayor,5))
        cv2.putText(self.frame,self.texto2,(10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)                    

        self.frame_final=cv2.resize(self.frame,(1280,960))    
        

        self.frame_final = cv2.cvtColor(self.frame_final, cv2.COLOR_BGR2RGBA)                
        self.photoFinal = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.frame_final))
        self.canvasVideo.create_image(0,0, image =self.photoFinal, anchor='nw')            
        
        
        self.root.after(self.delay, self.UpdateNeuroTester)    
# =============================================================================
        
class NeuroTrainer():
    
    def __init__(self):
        self.root=tk.Tk()
        self.root.configure(bg='gray13')
        self.root.title('Neurocode \ Data Base Cleaner')
        #extraccion de tamanio de la pantalla
        self.ws = self.root.winfo_screenwidth() # width of the screen
        self.hs = self.root.winfo_screenheight() # height of the screen
        # set the dimensions of the screen and where it is placed
        self.root.geometry('%dx%d+%d+%d' % (self.ws, self.hs, 0, 0))         
        self.root.minsize(int(self.ws/8),int(self.hs/8))       
## =============================================================================
##   Canvas Background
## =============================================================================
        self.canvasCero = tk.Canvas(self.root, width = self.ws*0.99, height = self.hs*0.99) # Neurocode Logo- windows 2 
        self.canvasCero.place(x=1, y=1)  
        self.back = cv2.imread('NeuroBackBlack.jpg')#direccion logo
        self.back=cv2.resize(self.back,(int(self.ws*0.99),int(self.hs*0.99)))    
        self.backPhoto = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.back))
        self.canvasCero.create_image(0, 0, image = self.backPhoto, anchor = tk.NW)                    
# =============================================================================
#   Canvas Logo
# =============================================================================
        self.canvasLogo = tk.Canvas(self.root, width = int(self.ws*0.15), height =int(self.hs*0.07),highlightthickness=0) # Neurocode Logo- windows 2 
        self.canvasLogo.place(x=int(self.ws*0.05), y=int(self.hs*0.05))  
        self.logo = cv2.imread('neurocodeNegro.jpg')#direccion logo
        self.logo=cv2.resize(self.logo,(int(self.ws*0.15),int(self.hs*0.07)))    
        self.photoLogo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(self.logo))
        self.canvasLogo.create_image(0, 0, image = self.photoLogo, anchor = tk.NW)     
# =============================================================================
   #  Botones y texto
# =============================================================================
        self.button1 = tk.Button(self.root, height=int(self.hs*0.005), width=int(self.ws*0.01),text="CREATE MODEL", bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.Training,font=("Berlin Sans FB", 30),state=tk.DISABLED)
        self.button1.place(x=int(self.ws*0.56),y=int(self.hs*0.38)) 

        tk.Button(self.root,height=int(self.hs*0.005), width=int(self.ws*0.01), text='IMAGE DATA BASE', bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.modelSelect,font=("Berlin Sans FB", 30)).place(x=int(self.ws*0.20),y=int(self.hs*0.38))        
        tk.Button(self.root,text='EXIT',command=self.exitAction,font=("Berlin Sans FB", 25),bg='gray13',relief='flat',fg='white').place(x=int(self.ws*0.885),y=int(self.hs*0.70))

        self.buttonPix30 = tk.Button(self.root, height=int(self.hs*0.0002), width=int(self.ws*0.002),text="30", bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.pix30,font=("Berlin Sans FB", 20),state=tk.DISABLED)
        self.buttonPix30.place(x=int(self.ws*0.56),y=int(self.hs*0.3)) 
        
        self.buttonPix50 = tk.Button(self.root, height=int(self.hs*0.0002), width=int(self.ws*0.002),text="50", bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.pix50,font=("Berlin Sans FB", 20),state=tk.DISABLED)
        self.buttonPix50.place(x=int(self.ws*0.61),y=int(self.hs*0.3)) 

        self.buttonPix80 = tk.Button(self.root, height=int(self.hs*0.0002), width=int(self.ws*0.002),text="80", bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.pix80,font=("Berlin Sans FB", 20),state=tk.DISABLED)
        self.buttonPix80.place(x=int(self.ws*0.66),y=int(self.hs*0.3)) 

        self.buttonPix100 = tk.Button(self.root, height=int(self.hs*0.0002), width=int(self.ws*0.002),text="100", bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.pix100,font=("Berlin Sans FB", 20),state=tk.DISABLED)
        self.buttonPix100.place(x=int(self.ws*0.71),y=int(self.hs*0.3)) 

        self.buttonPix150 = tk.Button(self.root, height=int(self.hs*0.0002), width=int(self.ws*0.002),text="150", bg='gray13',fg='white',relief='flat',borderwidth=2, command=self.pix150,font=("Berlin Sans FB", 20),state=tk.DISABLED)
        self.buttonPix150.place(x=int(self.ws*0.76),y=int(self.hs*0.3)) 

        self.textConsole = tk.Label(self.root, text="", bg='gray13',fg='white',font=("Berlin Sans FB", 20))
        self.textConsole.place(x=int(self.ws*0.60),y=int(self.hs*0.70)) 

# =============================================================================
# Parametros
# =============================================================================
        self.valordeLado=100#VALOR DEFAULT
        self.permittedExtension=".jpg"
        self.minimumAccuracy=0.95
        self.initialNeurons=1300
        self.stepsDownRate=1/8        
        self.directorioKey=dict()
        self.directorioString=""
        
# =============================================================================
# Loop
# =============================================================================
        self.root.mainloop()

    def exitAction(self):
        self.root.destroy()
        neurocodeGUI()
        
    def modelSelect(self):
        self.textConsole['text'] = ""
        self.filename = filedialog.askopenfilename()
        self.modelo1text=os.path.basename(self.filename)        
        self.directorio=os.path.dirname(self.filename)
        self.directorioPrincipal=os.path.abspath(os.path.join(self.directorio, '..'))     
        print("Database seleccionada: "+self.directorioPrincipal)
        self.buttonPix30['state'] = tk.NORMAL
        self.buttonPix50['state'] = tk.NORMAL
        self.buttonPix80['state'] = tk.NORMAL
        self.buttonPix100['state'] = tk.NORMAL
        self.buttonPix150['state'] = tk.NORMAL   
        self.button1['state'] = tk.NORMAL                         
        self.buttonPix100['fg'] = 'gray13' 
        self.buttonPix100['bg'] = 'white' 

    def pix30(self):
        self.buttonPix30['fg'] = 'gray13' 
        self.buttonPix30['bg'] = 'white' 
        self.buttonPix50['fg'] = 'white' 
        self.buttonPix50['bg'] = 'gray13' 
        self.buttonPix80['fg'] = 'white' 
        self.buttonPix80['bg'] = 'gray13' 
        self.buttonPix100['fg'] = 'white' 
        self.buttonPix100['bg'] = 'gray13' 
        self.buttonPix150['fg'] = 'white' 
        self.buttonPix150['bg'] = 'gray13' 
        self.valordeLado=30
        
    def pix50(self):
        self.buttonPix30['fg'] = 'white' 
        self.buttonPix30['bg'] = 'gray13'         
        self.buttonPix50['fg'] = 'gray13' 
        self.buttonPix50['bg'] = 'white' 
        self.buttonPix80['fg'] = 'white' 
        self.buttonPix80['bg'] = 'gray13' 
        self.buttonPix100['fg'] = 'white' 
        self.buttonPix100['bg'] = 'gray13' 
        self.buttonPix150['fg'] = 'white' 
        self.buttonPix150['bg'] = 'gray13' 
        self.valordeLado=50
        
    def pix80(self):
        self.buttonPix30['fg'] = 'white' 
        self.buttonPix30['bg'] = 'gray13'           
        self.buttonPix50['fg'] = 'white' 
        self.buttonPix50['bg'] = 'gray13' 
        self.buttonPix80['fg'] = 'gray13' 
        self.buttonPix80['bg'] = 'white' 
        self.buttonPix100['fg'] = 'white' 
        self.buttonPix100['bg'] = 'gray13' 
        self.buttonPix150['fg'] = 'white' 
        self.buttonPix150['bg'] = 'gray13' 
        self.valordeLado=80
        
    def pix100(self):
        self.buttonPix30['fg'] = 'white' 
        self.buttonPix30['bg'] = 'gray13'   
        self.buttonPix50['fg'] = 'white' 
        self.buttonPix50['bg'] = 'gray13' 
        self.buttonPix80['fg'] = 'white' 
        self.buttonPix80['bg'] = 'gray13' 
        self.buttonPix100['fg'] = 'gray13' 
        self.buttonPix100['bg'] = 'white' 
        self.buttonPix150['fg'] = 'white' 
        self.buttonPix150['bg'] = 'gray13' 
        self.valordeLado=100

    def pix150(self):
        self.buttonPix30['fg'] = 'white' 
        self.buttonPix30['bg'] = 'gray13'   
        self.buttonPix50['fg'] = 'white' 
        self.buttonPix50['bg'] = 'gray13' 
        self.buttonPix80['fg'] = 'white' 
        self.buttonPix80['bg'] = 'gray13' 
        self.buttonPix100['fg'] = 'white' 
        self.buttonPix100['bg'] = 'gray13' 
        self.buttonPix150['fg'] = 'gray13' 
        self.buttonPix150['bg'] = 'white' 
        self.valordeLado=150

    def Training(self):
               
        #Tamaño de imagen
        self.hsize=self.valordeLado
        self.vsize=self.valordeLado
# =============================================================================
# Start
# =============================================================================
        self.start = tm.time()
# =============================================================================
# Preprocesamiento de datos
# =============================================================================
        self.categorias=os.listdir(self.directorioPrincipal)
        if "Thumbs.db" in self.categorias:
           self.categorias.remove("Thumbs.db")
            
        self.foldersConCaracteristicas=os.listdir(self.directorioPrincipal)
        for i in range(len(self.foldersConCaracteristicas)):
            self.directorioKey[i] = self.foldersConCaracteristicas[i]
            
        self.classnume=len(self.categorias)
      
        self.valores=0    
        for category in self.categorias:
            self.path=os.path.join(self.directorioPrincipal,category)
            self.class_num=self.categorias.index(category)  
            
            self.valor=len(os.listdir(self.path))    
            self.valores=self.valores+self.valor
        
        self.trainingImageSet= np.zeros((self.valores,self.vsize,self.hsize,3))
        self.trainingLabelSet= np.zeros((self.valores,1))
        
        self.i=0      
        for category in self.categorias:
            self.path=os.path.join(self.directorioPrincipal,category)
            self.class_num=self.categorias.index(category) 
            
            for img in os.listdir(self.path):
                    self.file_extension = os.path.splitext(os.path.join(self.path,img))
                    
                    if self.file_extension[1]==self.permittedExtension:
                      self.img_array=cv2.imread(os.path.join(self.path,img))                
                      self.img_array=cv2.resize(self.img_array,(self.hsize,self.vsize))
                      self.img_array=self.img_array/np.amax(self.img_array)
                      self.trainingImageSet[self.i,:,:,:]=self.img_array
                      self.trainingLabelSet[self.i,:]=self.class_num
                      self.i=self.i+1
    
        self.y_train = keras.utils.to_categorical(self.trainingLabelSet, num_classes=self.classnume)
        # =============================================================================
        # Distribucion de datos para entrenamiento
        # =============================================================================
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.trainingImageSet, self.y_train, test_size=0.10, random_state=101)
        train_test_split(self.y_train, shuffle=False)#se evita combinar
        # =============================================================================
        # Disenio del modelo
        # =============================================================================
        model = Sequential()
        # =============================================================================
        # Convolution Layers
        # =============================================================================
        
        model.add(Conv2D(8, (3, 3), padding='same',input_shape=(self.vsize,self.hsize,3),strides = (1,1),activation='elu'))
        model.add(BatchNormalization())
        
        model.add(Conv2D(16, (3, 3),strides = (1,1),activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(32, (3, 3),strides = (1,1),activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(64, (3, 3), strides = (1,1),activation='elu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(128, (2, 2),strides = (1,1),activation='elu'))
        
        # =============================================================================
        # Fully conected layers
        # =============================================================================
        model.add(Flatten())
        
        self.difference=self.initialNeurons-self.classnume
        
        for i in range(100):
          model.add(Dense(self.initialNeurons,activation='elu'))
          model.add(Dropout(0.2))
          self.initialNeurons=round(self.initialNeurons*self.stepsDownRate)
          if self.initialNeurons <= self.classnume:
               break
        
        model.add(Dense(self.classnume, kernel_initializer='normal',activation='softmax'))
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
        #    zoom_range=-0.2,
            shear_range=0.2)
        
        datagen.fit(self.X_train)

        # =============================================================================
        # stopping function for training
        # =============================================================================
        stopping=EarlyStopping(monitor='acc', min_delta=0.001, patience=6, verbose=0)
        # ===========================================================================
        
        #Model Training
        # =============================================================================
        batchSize=30
        model.fit_generator(datagen.flow(self.X_train, self.Y_train, batch_size=batchSize),# fits the model on batches with real-time data augmentation:
                            steps_per_epoch=len(self.X_train)/batchSize,epochs=1000,callbacks=[stopping])
        # =============================================================================
        # Entrenamiento del modelo
        # =============================================================================
        #model.fit(X_train, Y_train, batch_size=32, epochs=100)
        # =============================================================================
        # Evaluacion del modelo
        # =============================================================================
        self.score = model.evaluate(self.X_test, self.Y_test, batch_size=batchSize)
        print("Metrics(test loss & Test Accuracy): ")
        print(self.score)
        self.end=tm.time()
        self.horas=((self.end-self.start)/60)/60
        print('Tiempo de procesamiento [Hr]:')
        print(round(self.horas,5))
        
        # =============================================================================
        # Guardar modelo
        # =============================================================================
        if self.score[1]>self.minimumAccuracy:
          self.scoreString=str(self.score[1])
          self.scoreLoss=str(self.score[0])
          self.output = tm.strftime("%d-%m-%Y-%H-%M-%S")   

            # =============================================================================
            #creacion de folder para cuarentena
          self.storageFolder="C:/TEMP/NEURO MODELS"        
          self.isdir = os.path.isdir(self.storageFolder)         
          if self.isdir ==False:
               os.mkdir(self.storageFolder)  

          for i in range(len(self.directorioKey)):
                self.directorioString=self.directorioString+str(i)+":"+self.directorioKey[i]+","
          self.directorioString=self.directorioString
	      
          self.textoIndex='modelo_'+os.path.basename(self.directorioPrincipal)+'_'+"LXL"+str(self.hsize)+'x'+str(self.vsize)+'_Loss_'+self.scoreLoss[2:5]+'_Acc_'+self.scoreString[2:5]+'_T_'+self.output+"_"+(socket.gethostname())+'.model'
          file= open(self.storageFolder+'/'+"Directorios_Creados_Neuro.txt","a+")  
          
          file.write(self.textoIndex+"%"+self.directorioString)
          file.close()
        
          os.startfile(self.storageFolder)                        

            
          model.save(self.storageFolder+'/'+self.textoIndex)     
          print('Modelo guardado')
          self.textConsole['text'] = "MODEL COMPLETED"

        else:
                        
            print('El modelo no cumple con los criterios de precision minima requerida')
            self.textConsole['text'] = "MODEL DOESN'T MEET MINIMUM QUALITY REQUIREMENTS"

        self.button1['state'] = tk.DISABLED          
        self.buttonPix50['state'] = tk.DISABLED  
        self.buttonPix80['state'] = tk.DISABLED  
        self.buttonPix100['state'] = tk.DISABLED  
        self.buttonPix150['state'] = tk.DISABLED  
        os.startfile(self.storageFolder)                        
# =============================================================================
#              Main call
# =============================================================================
#DataBaseCreator() 
#NeuroTesterClass()        
neurocodeGUI()
#Ventana()
