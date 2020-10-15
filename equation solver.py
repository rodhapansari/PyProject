import cv2
import threading
import numpy
import pytesseract

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

def f1(frame,oldframe,n,m):
    #1nd Quad 
    for i in range(0,int(n/2)):
        for j in range(int(m/2),m):
            if(frame[i,j,0] > 150 and frame[i,j,1] > 150 and frame[i,j,2] > 150 ):
                frame[i,j] = oldframe[i,j]
            
            #if(i> 150 and i<200 and j>150 and j<200 ):
            #    frame[i,j] = [255,255,255]

def f2(frame,oldframe,n,m):
    #2nd Quad         
    for i in range(0,int(n/2)):
        for j in range(0,int(m/2)):
            if(frame[i,j,0] > 150 and frame[i,j,1] > 150 and frame[i,j,2] > 150 ):
                frame[i,j] = oldframe[i,j]
            
            #if(i> 150 and i<200 and j>150 and j<200 ):
            #    frame[i,j] = [255,255,255]

def f3(frame,oldframe,n,m):
    #3nd Quad 
    for i in range(int(n/2),n):
        for j in range(0,int(m/2)):
            if(frame[i,j,0] > 150 and frame[i,j,1] > 150 and frame[i,j,2] > 150 ):
                frame[i,j] = oldframe[i,j]
    


def f4(frame,oldframe,n,m):
    #4nd Quad 
    for i in range(int(n/2),n):
        for j in range(int(m/2),m):
            if(frame[i,j,0] > 150 and frame[i,j,1] > 150 and frame[i,j,2] > 150 ):
                frame[i,j] = oldframe[i,j]
            
            #if(i> 150 and i<200 and j>150 and j<200 ):
            #    frame[i,j] = [255,255,255]
            
            #if(i> 150 and i<200 and j>150 and j<200 ):
            #    frame[i,j] = [255,255,255]
    

frame= numpy.empty(3, dtype=int)
def read(cam): 
    retq, frame = cam.read()
    

img_counter = 0
i=0

while(i<100):
    retq, oldframe = cam.read()
    i+=1
    print(i)

print("old capture")
    
while True:
    ret, img = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    

    
    r = threading.Thread(target=read, args=(cam,))
    r.start()
    
    #n=480,m=640
    #def f4(frame,oldframe,n,m):

    
    #Alternatively: can be skipped if you have a Blackwhite image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(gray)
    
    kernel = np.ones((2, 1), np.uint8)
    img = cv2.erode(gray, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    out_below = pytesseract.image_to_string(img)
    print("OUTPUT:", out_below)
                
    
    
    
    
    
    
            
    cv2.imshow("test", frame)
                
    
    

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    

cam.release()

cv2.destroyAllWindows()
