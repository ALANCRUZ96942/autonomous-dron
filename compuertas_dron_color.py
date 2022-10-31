from re import S
from djitellopy import Tello
import cv2
import numpy as np
import time
 
######################################################################
width = 640  # WIDTH OF THE IMAGE
height = 480  # HEIGHT OF THE IMAGE
centro =50
centro2 = 70
######################################################################
 
startCounter =0
 
# CONNECT TO TELLO
me = Tello()
me.connect()
me.for_back_velocity = 0
me.left_right_velocity = 0
me.up_down_velocity = 0

me.yaw_velocity = 0
me.speed = 0
 
 
 
print(me.get_battery())
 
me.streamoff()
me.streamon()
######################## 
 
frameWidth = width
frameHeight = height
# cap = cv2.VideoCapture(1)
# cap.set(3, frameWidth)
# cap.set(4, frameHeight)
# cap.set(10,200)
 
 
global imgContour
global dir
def empty(a):
    pass
 
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",0,179,empty)
cv2.createTrackbar("HUE Max","HSV",11,179,empty)
cv2.createTrackbar("SAT Min","HSV",140,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",89,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",80,255,empty)
cv2.createTrackbar("Threshold2","Parameters",171,255,empty)
cv2.createTrackbar("Area","Parameters",250,30000,empty)
 
 
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
 
def getContours(img,imgContour):
    global dir
    contours, hierarchy = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    gate =  0
   

    for cnt in contours:

        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        


        #cv2.drawContours(imgContour, cnt, -1, (255, 0, 255),3)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
        x , y , w, h = cv2.boundingRect(approx)   
        aspRatio = w/float(h)
        
        
        if(aspRatio >0.95 and aspRatio <1.05 and hierarchy[0][gate][3] != -1 and area > areaMin):

            for point in approx:
                point = point[0] # drop extra layer of brackets
                center = (int(point[0]), int(point[1]))
                cv2.putText(imgContour, "coordinate: " + str(center[0])+ " "+str(center[1]), (center[0] + 10, center[1] +10), cv2.FONT_HERSHEY_COMPLEX, .5,
                (255, 0, 0),2)
                cv2.circle(imgContour, center, 4, (150, 200, 0), -1)
            
            print(len(approx))
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 1)
            cv2.putText(imgContour, "Compuerta: " + str(gate), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0),2)


            M = cv2.moments(cnt)



            if M["m00"] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                # set values as what you need in the situation
                cx, cy = 0, 0


            cv2.circle(imgContour,(cx,cy),7,(0,255,23),-1)
            cv2.circle(imgContour,(cx,cy),7,(0,255,23))
            

            gate =+ 1
            print(h)
            if(h < frameHeight-100):
                if (cx < int(frameWidth/2)-centro):
                    cv2.putText(imgContour, "Izquierda" , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                    dir = 1
                elif (cx > int(frameWidth / 2) + centro):
                    cv2.putText(imgContour, "Derecha", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                    dir = 2
                elif (cy < int(frameHeight / 2) - centro):
                    cv2.putText(imgContour, "Arriba", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                    dir = 3
                elif (cy > int(frameHeight / 2) + centro):
                    cv2.putText(imgContour, "Abajo", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 3)
                    dir = 4
                elif(cx < int(frameWidth/2)+centro and cx > int(frameWidth / 2) - centro and cy < int(frameHeight / 2) + centro and cy > int(frameHeight / 2) - centro):
                    cv2.putText(imgContour, "Adelante", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 3)
                    dir = 5
                cv2.line(imgContour, (int(frameWidth/2),int(frameHeight/2)), (cx,cy),
                        (0, 0, 255), 3)
            else:
                cv2.putText(imgContour, "Pasando compuerta" , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
                dir = 6

        gate = gate+1
            



 
while True:
 
    # GET THE IMAGE FROM TELLO
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))
    imgContour = img.copy()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
    h_min = cv2.getTrackbarPos("HUE Min","HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min = cv2.getTrackbarPos("SAT Min", "HSV")
    s_max = cv2.getTrackbarPos("SAT Max", "HSV")
    v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
    v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
 
 
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHsv,lower,upper)
    result = cv2.bitwise_and(img,img, mask = mask)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
 
    imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
    imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    kernel = np.ones((4, 4))
    imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    getContours(imgDil, imgContour)
    #display(imgContour)
    stack = stackImages(0.9, ([img, result], [imgDil, imgContour]))
    cv2.imshow('Dron vision', stack)
    ################# FLIGHT
    
    if cv2.getWindowProperty('Dron vision', cv2.WND_PROP_VISIBLE) >= 1 and startCounter == 0:
        me.takeoff()
        while(me.get_height() < 90 ):
            me.send_rc_control(0,0,30,0)
        startCounter = 1
 
    if(me.get_height() >= 80 ):
        if dir == 1:
            me.left_right_velocity = -20; me.for_back_velocity = 0;me.up_down_velocity = 0; me.yaw_velocity =  0
        elif dir == 2:
            me.left_right_velocity = 20; me.for_back_velocity = 0;me.up_down_velocity = 0; me.yaw_velocity = 0 
        elif dir == 3:
            me.left_right_velocity = 0; me.for_back_velocity = 0;me.up_down_velocity = 30; me.yaw_velocity = 0    
        elif dir == 4:
            me.left_right_velocity = 0; me.for_back_velocity = 0;me.up_do
            wn_velocity = -30; me.yaw_velocity = 0 
        elif dir == 5:
            me.left_right_velocity = 0; me.for_back_velocity = 30;me.up_down_velocity = 0; me.yaw_velocity = 0 
        elif dir == 6:
            me.left_right_velocity = 0; me.for_back_velocity = 40;me.up_down_velocity = -20; me.yaw_velocity = 0
            me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
        else:
            me.left_right_velocity = 0; me.for_back_velocity = 15;me.up_down_velocity = 0; me.yaw_velocity = 0
    # SEND VELOCITY VALUES TO TELLO
    else:
        me.send_rc_control(0,0,30,0)



    if me.send_rc_control:
         me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)
    

    print(dir)
    """
    stack = stackImages(0.9, ([img, result], [imgDil, imgContour]))
    cv2.imshow('Dron vision', stack)"""
 
    if cv2.waitKey(1) & 0xFF == ord('q'):

        me.land()
        break
 
# cap.release()
cv2.destroyAllWindows()