

import numpy as np
import math
import cv2

def WarpImage(frame):
    ax,bx=10.0,100
    ay,by=20.0,120
    img=np.zeros(frame.shape,dtype=frame.dtype)
    rows,cols=img.shape[:2]
    
    for i in range(rows):
        for j in range(cols):
            offset_x=int(ax*math.sin(2*math.pi*i/bx))
            offset_y=int(ay*math.cos(2*math.pi*j/by))
            if i+offset_y<rows and j+offset_x<cols:
                img[i,j]=frame[(i+offset_y)%rows,(j+offset_x)%cols]
            else:
                img[i,j]=0
    return img

def equalizeHistColor(frame):
    # equalize the histogram of color image
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # convert to HSV
    img[:,:,2] = cv2.equalizeHist(img[:,:,2])     # equalize the histogram of the V channel
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)   # convert the HSV image back to RGB format


# start video capture
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read() 
    frame=cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
    
    # Our operations on the frame come here 
    if ret==1:
        #img = WarpImage(frame)
        img = equalizeHistColor(WarpImage(frame))
    else:
        img = equalizeHistColor(frame)
        
    # Display the resulting image
    cv2.imshow('Warped',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

