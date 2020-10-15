import cv2

import numpy as np
import time
import pyautogui
import math



import pandas as pd



df1 = pd.read_csv("datafile (2) (1).csv")
img = cv2.imread('plot.png')

import matplotlib.pyplot as plt 

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


# line 1 points
y1 = df1.loc[1][1:]
x1 = range(len(df1.loc[1][1:]))
# plotting the line 1 points 
plt.plot(x1, y1, label = df1.loc[1][0])
# line 2 points
y2 = df1.loc[2][1:]
x2 = range(len(df1.loc[2][1:]))
# plotting the line 2 points 
plt.plot(x2, y2, label = df1.loc[2][0])
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title of the current axes.
plt.title('Two or more lines on same plot with suitable legends ')
# show a legend on the plot
plt.legend()
plt.savefig('plot.png') 


#plt.savefig('plot.png')

print()
#https://www.gameflare.com/online-game/table-tennis-pro/
vs = cv2.VideoCapture(0)

ub = np.array([171,113,45])
lb = np.array([87,0,0])

def translate(sensor_val, in_from, in_to, out_from, out_to):
    out_range = out_to - out_from
    in_range = in_to - in_from
    in_val = sensor_val - in_from
    val=(float(in_val)/in_range)*out_range
    out_val = out_from+val
    return out_val

def nothing(x):
    pass

cv2.namedWindow("Control")

cv2.createTrackbar("L-H","Control",0,180,nothing)
cv2.createTrackbar("L-S","Control",0,255,nothing)
cv2.createTrackbar("L-V","Control",0,255,nothing)
cv2.createTrackbar("U-H","Control",0,180,nothing)
cv2.createTrackbar("U-S","Control",0,255,nothing)
cv2.createTrackbar("U-V","Control",0,255,nothing)
x,y = pyautogui.size()
print(x,y)

_,f = vs.read()
print(f.shape)
#x ->width
#y->height

height, width,_ = f.shape
last_cx = 0
last_cy = 0

while True:
    
    _,f = vs.read()
    #f = cv2.resize(f, (x,y), interpolation = cv2.INTER_AREA)
    f = cv2.flip(f,1)
    cpy = f.copy()
    
    k=cv2.imread('yello.png')
    
    
    hsv = cv2.cvtColor(cpy, cv2.COLOR_BGR2HSV)
    
    lh = cv2.getTrackbarPos("L-H","Control")
    ls = cv2.getTrackbarPos("L-S","Control")
    lv = cv2.getTrackbarPos("L-V","Control")
    uh = cv2.getTrackbarPos("U-H","Control")
    us = cv2.getTrackbarPos("U-S","Control")
    uv = cv2.getTrackbarPos("U-V","Control")
    
    
    mask1 = cv2.inRange(hsv,(lh,ls,lv),(uh,us,uv))
    #mask1 = cv2.inRange(hsv,(88,223,0),(180,255,255))
    

    #mask1 = cv2.inRange(hsv,lb,ub)
    
    #=======================

    
    mx,my = pyautogui.position()

    
    #=======================
    cv2.putText(k,"Plot 1", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cv2.rectangle(k, (0, 0), (200, 100), (255,0,0), 2)
    cv2.rectangle(k, (0, 100),(200,200), (255,255,0), 2)
    cv2.putText(k,"Close Plot 1", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
    cnts, hierarchy = cv2.findContours(mask1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        pass
    else:
        segmented = max(cnts, key=cv2.contourArea)
        M = cv2.moments(segmented)
        
        try:
            cx = round(M['m10'] / M['m00'])
            cy = round(M['m01'] / M['m00'])
            print(cx,cy)
            cv2.circle(k, (cx, cy), 5, (255, 0, 0), -1)
            if((cx>0 and cx<200) and (cy>0 and cy<100)):
            #pyautogui.moveTo(cx, cy, duration=0.01)
                
                cv2.imshow('Plot',img)
            if((cx>0 and cx<200) and (cy>100 and cy<200)):
            #pyautogui.moveTo(cx, cy, duration=0.01)
                
                
                cv2.destroyWindow('Plot')   
                
        except:
            pass
        '''
        for p in segmented:
            cx += p[0][0]
            cy += p[0][1]
            #cv2.drawContours(cpy, [p], -1, (0, 255, 0), 2)
        cx = int(cx/len(segmented))
        cy = int(cy/len(segmented))
        '''
        #cv2.circle(cpy, (cx, cy), 7, (255, 255, 255), -1)

        
        
            
        
        

    
    cv2.imshow("nf",k)
    cv2.imshow("new mask", cv2.resize(mask1, (300,300), interpolation = cv2.INTER_AREA))
    
    
    
    

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        time.sleep(5)
    
    elif k%256 == 32:
        # ESC pressed
        print("Space hit, closing...")
        break

    
    



cv2.destroyAllWindows()

































