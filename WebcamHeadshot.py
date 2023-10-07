# -*- coding: utf-8 -*-
"""
Alex Johnson-Buck

WebcamHeadshot
View webcam feed, apply filters and take headshot

Recognizes face in field of view using Haar cascades and takes a headshot upon pressing c key.

Apply blur and other filters 

"""

import cv2
import numpy as np
import sys
import copy

#%%-------------------------------------------------------------------------------------
# Function and Class Definitions

def cascadeClassifier(modelpath):
    featureClassifier = cv2.CascadeClassifier()
    featureClassifier.load(modelDir+modelFile)
    return featureClassifier

def annotateImage(image,message,origin):
    fontScale = message.scale
    lineWidth = message.lineWidth
    text = message.text
    color = message.color
    image = cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 
    fontScale, color, lineWidth, cv2.LINE_AA)

def padFeatureBounds(features): # pads feature boundaries by 50%
    for (x,y,w,h) in features:
        x = x-w//4
        y = y-h//4
        h = h*6//4
        w = w*6//4
    return x,y,w,h

def detectFeatures(frame,featureClassifier,minSize):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    features = featureClassifier.detectMultiScale(frame_gray,1.1,3,0,minSize)
    if np.size(features,0) > 0:
        x,y,w,h = padFeatureBounds(features)
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2, cv2.LINE_AA)
    return frame, features

def capFeature(cap,features,title,fName,displayMode):
    x,y,w,h = padFeatureBounds(features)
    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = applyDisplayMode(frame,displayMode,cannyThresh)
    featureCap = frame[y:y+h,x:x+w]
    featureCap = cv2.flip(featureCap,1)
    cv2.imshow(title, featureCap)
    cv2.imwrite(fName, featureCap)
    print("Saving headshot to file...")

    
def tryCapFace(cap,faces,title,fName,displayMode):
    message = msg()
    if np.size(faces,0) > 1:
        message.text = 'Error: Multiple faces detected.  Cannot capture headshot.  Please try again.'
        message.color = (0,0,255)
    elif np.size(faces,0) < 1:
        message.text = 'Error: No faces detected.  Cannot capture headshot.  Please try again.'
        message.color = (0,0,255)
    else:
        try:
            capFeature(cap,faces,title,fName,displayMode)
            message.text = "Headshot successfully captured and saved as headshot.png."
            message.color = (255,255,255)
        except:
            cv2.destroyWindow(title)
            message.text = "Error: could not capture face.  Please try again.  Make sure your face is centered in the image."
            message.color = (0,0,255)
    return message

class msg:
    text = '[default text]'
    color = (255,255,255)
    scale = 0.5
    lineWidth = 1
    
def applyDisplayMode(image,displayMode,cannyThresh):
    if displayMode == 1: # blur
        image = cv2.blur(image,(11,11)) # blur with square kernel
        # image = cv2.bilateralFilter(image,11,25,5) # bilateral filtering
    elif displayMode == 2: # canny
        image_orig = copy.deepcopy(image)    
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.bilateralFilter(image,11,25,5) # bilateral filtering
        # image = cv2.GaussianBlur(image,(11,11),3) # gaussian filtering
        # image = cv2.blur(image,(11,11))          # blur with square kernel
        image = cv2.Canny(image,cannyThresh[0],cannyThresh[1])
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        image = np.uint8(np.clip(np.double(image_orig) - np.double(image),0,255)) # cartoony overlay
    elif displayMode == 3: # Gaussian blur
        image = cv2.GaussianBlur(image,(11,11),4)
    return image

#%% Main Body of Code

# Symbolic constants for image display modes
NORMAL = 0
BLUR   = 1
CANNY  = 2
GAUSSIAN = 3


modelFile = "haarcascade_frontalface_alt.xml"
modelDir = ".\\haarcascades\\"

faceClassifier = cascadeClassifier(modelDir+modelFile)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: camera opening unsuccessful.  Quitting.")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

msg1 = msg()
msg1.text = 'Commands: c = capture headshot, n = unfiltered, b = apply blur, g = Gaussian blur, e = Canny edge detection, q = quit'
msg1.color = (0,255,255)

msg2 = msg()
msg2.text = ''
msg2.color = (0,0,255)

# copyrightmsg = msg()
# copyrightmsg.text = 'Copyright Alex Johnson-Buck, 2022'

cannyThresh = [10,100]

displayMode = NORMAL

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame,1)
    
    frame = applyDisplayMode(frame,displayMode,cannyThresh)
    
    frame, faces = detectFeatures(frame,faceClassifier,(100,100))
    
    # annotateImage(frame,copyrightmsg,(50,50))
    annotateImage(frame,msg1,(50,670))
    annotateImage(frame,msg2,(50,620))
    cv2.imshow("Camera feed", frame)

    key = cv2.waitKey(1)
    if key == ord('c') or key == ord('C'):
        if displayMode != CANNY:
            msg2 = tryCapFace(cap,faces,"Face capture","Headshot.png",displayMode)
    elif key == ord('b') or key == ord('B'):
        displayMode = BLUR
    elif key == ord('e') or key == ord('E'):
        displayMode = CANNY
    elif key == ord('[') and displayMode == CANNY and cannyThresh[0] > 0:   # Use [,],-,= keys to adjust Canny thresholds
        cannyThresh[0] -= 10
    elif key == ord(']') and displayMode == CANNY and cannyThresh[0] < cannyThresh[1]-10:
        cannyThresh[0] += 10
    elif key == ord('-') and displayMode == CANNY and cannyThresh[1] > cannyThresh[0]+10:
        cannyThresh[1] -= 10
    elif key == ord('=') and displayMode == CANNY and cannyThresh[1] < 245:
        cannyThresh[1] += 10
    elif key == ord('n') or key == ord('N'):
        displayMode = NORMAL
    elif key == ord('g') or key == ord('G'):
        displayMode = GAUSSIAN
    elif key == ord('q') or key == ord('Q'):
        print("Quitting...")
        break
    
cap.release()

cv2.destroyAllWindows()