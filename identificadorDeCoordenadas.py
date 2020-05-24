# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:51:27 2019

@author: E803562 Victor Rodriguez
"""
#Codigo para obtener seccciones de imagenes y realizar base de datos
import tkinter as tk
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import filedialog

class App:
      def __init__(self, master,filename):
         self.window = master
         self.filename = filename
       
         # open video source (by default this will try to open the computer webcam)
         self.path=self.filename
         self.resized = cv2.imread(self.path)          
         self.tamanioOriginal=self.resized.shape
                  
         # Create a canvas that can fit the above video source size
         self.canvas = tk.Canvas(master, width = self.tamanioOriginal[1], height = self.tamanioOriginal[0])
         self.canvas.place(x=0, y=0)
 
         # After it is called once, the update method will be automatically called every delay milliseconds
         self.delay = 2
         self.update()
 
         self.window.mainloop()
 
            
      def callback (self,event):
          self.xcord=event.x
          self.ycord=event.y
          print("("+str(self.xcord)+','+str(self.ycord)+')')  
                             
      def update(self):
         # Get a frame from the video source
         frame=self.resized.copy()
         self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
         self.canvas.create_image(0, 0, image = self.photo, anchor = tk.NW)
         self.canvas.bind("<Button-1>", self.callback)
         self.window.after(self.delay, self.update)
 
# =============================================================================
# Main window 
# =============================================================================
master = tk.Tk()
master.minsize(100,100)
w = 660 # width for the Tk root
h = 500 # height for the Tk root

# get screen width and height
ws = master.winfo_screenwidth() # width of the screen
hs = master.winfo_screenheight() # height of the screen

# calculate x and y coordinates for the Tk root window
x = (ws/8) - (w/8)
y = (hs/8) - (h/8)

# set the dimensions of the screen 
# and where it is placed
master.geometry('%dx%d+%d+%d' % (w, h, x, y))

#master.iconbitmap(default='HCMO.ico')
master.title("Coordinates selection TK")
master.configure(background='grey')
# =============================================================================
# Actualizar la carpeta donde estan las imagenes (misma que se usara para guardar los recortes)

def browse_button():
    filename = filedialog.askopenfilename()
    return filename

filename=browse_button()
print(filename)
# =============================================================================
# app call
# =============================================================================
App(master,filename)