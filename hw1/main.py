import sys
import cv2
from util import Utils
from time import sleep

if __name__ == "__main__":
    utlis = Utils()


    if len(sys.argv) < 2:
        cap = cv2.VideoCapture(0)
        # Video
        ret, frame = cap.read()
        pass
    else:
        # numpy.ndarry , (1080, 1920, 3)
        frame = cv2.imread(sys.argv[1])

    while(True):            
        if len(sys.argv) < 2:
            cap = cv2.VideoCapture(0)
            # Video
            ret, frame = cap.read()
            pass
        else:
            # numpy.ndarry , (1080, 1920, 3)
            frame = cv2.imread(sys.argv[1])


        rawPressedKey = cv2.waitKey(1)

        if rawPressedKey == ord('p'):
            print('exit')
            break

        frame = utlis.processKey(frame,rawPressedKey)
        frame = cv2.resize(frame, (960, 540))                  # Resize image
        cv2.imshow('output',frame)

    cv2.destroyAllWindows()