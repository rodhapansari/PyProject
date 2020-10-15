import pandas as pd
import numpy as np
import cv2


df1 = pd.read_csv("datafile (2) (1).csv")


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
plt.xlabel('Year')
# Set the y axis label of the current axis.
plt.ylabel('Production')
# Set a title of the current axes.
plt.title('Four Year Production Plot 2006-2010 ')
# show a legend on the plot
plt.legend()
plt.savefig('plot.png') 


#plt.savefig('plot.png')

print()



img = cv2.imread('plot.png')
cv2.imshow('Plot',img)

