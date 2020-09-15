import numpy as np
import cv2
import math
from time import sleep

class Utils:
    def __init__(self):
        self.pressedKey = ord('i')
        self.current_channel = 0
        # ksize 
        self.ksize = (10, 10) 
        self.options = {
            ord('i') : self.doNothing,
            ord('g') : self.convertToGrayscaleUsingOpenCV,
            ord('q') : self.convertToGrayscale,
            ord('c') : self.cycleThroughChannels,
            ord('s') : self.blurImageUsingOpenCV,
            ord('d') : self.blurImage,
            ord('x') : self.computeXDerivative,
            ord('y') : self.computeYDerivative,
            ord('m') : self.computeDerivative,
            ord('v') : self.drawGradient,
            ord('r') : self.rotateImage,
        }


        self.windowsOn = False
        self.helpText = [
            "[i] - reload the original image (i.e. cancel any previous processing)",
            "[w] - save the current (possibly processed) image into the file out.jpg",
            "[g] - convert the image to grayscale using the openCV conversion function.",
            "[q] - convert the image to grayscale using your implementation of conversion function.",
            "[c] - cycle through the color channels of the image.",
            "[s] - convert the image to grayscale and smooth it using the openCV function.",
            "[d] - convert the image to grayscale and smooth it using my function.",
            "[x] - convert the image to grayscale and perform convolution with an x derivative filter.",
            "[y] - convert the image to grayscale and perform convolution with a y derivative filter.",
            "[m] - show the magnitude of the gradient normalized to the range [0,255].",
            "[v] - convert the image to grayscale and plot the gradient vectors of the image every N pixels.",
            "[r] - convert the image to grayscale and rotate it using an angle of Q degrees.",
            "[h] - Display this help window.",
            "[p] - Quit the program."
        ]

    def doNothing(self,frame):
        return frame

    def saveImage(self,frame):
        cv2.imwrite('out.jpg',frame)
        return frame

    def convertToGrayscaleUsingOpenCV(self,frame):
        assert len(frame.shape) == 3
        
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        return frame

    def convertToGrayscale(self,frame):
        assert len(frame.shape) == 3

        a , b, c = 0.122 , 0.365, 0.277
        frame = a * frame[:,:,0] + b * frame[:,:,1] + c * frame[:,:,2]
        
        return np.uint8(frame)
    
    def cycleThroughChannels(self,frame):
        result = np.zeros_like(frame)
        result[:,:,self.current_channel] = frame[:,:,self.current_channel]
        self.updateCycleChannel()
        return result

    def blurImageUsingOpenCV(self,frame):
        # Using cv2.blur() method  
        frame = cv2.blur(frame, self.ksize)  

        return frame

    def placeHolder(self, x):
        pass

    def blurImage(self,frame):
        # Average blur
        if not self.windowsOn:
            cv2.namedWindow("Trackbar")
            cv2.createTrackbar("Blur", "Trackbar", 0, 10, self.placeHolder)
            self.windowsOn = True

        kernel_size =  cv2.getTrackbarPos('Blur','Trackbar') 
        if kernel_size != 0:
            kernel = np.ones((kernel_size,kernel_size)) / (kernel_size**2)  
            frame = cv2.filter2D(frame,-1,kernel)
        
        # frame = cv2.resize(frame, (960, 540))                  # Resize image
        # cv2.imshow('output',frame)

        return frame

    def computeXDerivative(self,frame):
        '''
        convert the image to grayscale and perform convolution with an x derivative filter.
        '''
        # Convert 2 grayscale
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Sober derivative
        xDerivativeKernel = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ])
        frame = cv2.filter2D(frame, cv2.CV_16S, xDerivativeKernel)
        # Normalize
        frame = 1.0 * (frame - frame.min()) / (frame.max() - frame.min()) * 255

        return np.uint8(frame)
        
    def computeYDerivative(self,frame):
        '''
        convert the image to grayscale and perform convolution with an x derivative filter.
        '''
        # Convert 2 grayscale
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Sober derivative
        yDerivativeKernel = np.array([
            [-1 ,-2 ,1],
            [0  ,0  ,0],
            [1  ,2  ,1]
        ])
        frame = cv2.filter2D(frame, cv2.CV_16S, yDerivativeKernel)
        # Normalize
        frame = 1.0 * (frame - frame.min()) / (frame.max() - frame.min()) * 255

        return np.uint8(frame)

    def computeDerivative(self,frame, return_deri = False):
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        xDerivativeKernel = np.array([
            [-1,0,1],
            [-2,0,2],
            [-1,0,1]
        ])
        xDerivative = cv2.filter2D(frame, cv2.CV_16S, xDerivativeKernel)

        yDerivativeKernel = np.array([
            [-1 ,-2 ,1],
            [0  ,0  ,0],
            [1  ,2  ,1]
        ])
        yDerivative = cv2.filter2D(frame, cv2.CV_16S, yDerivativeKernel)

        frame = 0.5 * np.abs(xDerivative) + 0.5 * np.abs(yDerivative)
        frame = 1.0 * (frame - frame.min()) / (frame.max() - frame.min()) * 255

        if return_deri:
            return frame, xDerivative, yDerivative
        else:
            return frame

    def drawGradient(self,frame):
        '''
        convert the image to grayscale and plot the gradient vectors of the image every N pixels.
        '''
        if not self.windowsOn:
            cv2.namedWindow("Trackbar")
            cv2.createTrackbar("Step", "Trackbar", 0, 10, self.placeHolder)
            self.windowsOn = True
        # Convert 2 gray scale
        frame, xDerivative, yDerivative = self.computeDerivative(frame,True)

        step_size = cv2.getTrackbarPos("Step", "Trackbar") * 5 + 20
        gradient = np.zeros_like(frame)
        for x in range(step_size // 2,xDerivative.shape[0],step_size):
            for y in range(step_size // 2,yDerivative.shape[1],step_size):
                angle = math.atan2(xDerivative[x,y], yDerivative[x,y])
                newX = int(x + step_size*math.cos(angle))
                newY = int(y + step_size*math.sin(angle))
                cv2.arrowedLine(frame, (y, x), (newY, newX), (0, 0, 0))

        return np.uint8(frame)

    def rotateImage(self,frame):
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not self.windowsOn:
            cv2.namedWindow("Trackbar")
            cv2.createTrackbar("Degree", "Trackbar", 0, 180, self.placeHolder)
            self.windowsOn = True

        angle =  cv2.getTrackbarPos('Degree','Trackbar')

        # grab the dimensions of the image and then determine the
        # center
        (h, w) = frame.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        # perform the actual rotation and return the image
        frame = cv2.warpAffine(frame, M, (nW, nH))
        return frame

    def processKey(self,frame,rawPressedKey):
        self.pressedKey = rawPressedKey 

        if self.pressedKey in self.options:
            frame = self.options[self.pressedKey](frame)
        else:
            if self.pressedKey == ord('h'):
                self.displayHelp()
        frame = cv2.resize(frame, (960, 540))                  # Resize image
        cv2.imshow('output',frame)
        sleep(0.3)
        return frame

    def updateCycleChannel(self):
        if self.current_channel < 2:
            self.current_channel +=1
        else:
            self.current_channel = 0
        
    def showHelp(self):
        for item in self.helpText:
            print(item)

    def displayHelp(self):
            cv2.namedWindow("Help")
            helpFrame = np.zeros((400, 1280))
            originX = 25
            originY = 25

            for i in range(len(self.helpText)):
                cv2.putText(helpFrame, self.helpText[i], (originX, originY), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)	
                textSize = cv2.getTextSize(self.helpText[i], cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 1)
                originY += textSize[1] + 15
            
            cv2.imshow("Help", helpFrame)