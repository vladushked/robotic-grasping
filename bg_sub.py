from __future__ import print_function
import cv2 as cv
import argparse

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='KNN')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [create]

## [capture]
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

init_frame = None

while True:
    ret, init_frame = capture.read()
    init_frame = cv.cvtColor(init_frame, cv.COLOR_BGR2GRAY)
    cv.imshow('Frame', init_frame)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


while True:
    ret, frame = capture.read()
    if frame is None:
        break

    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    

    ## [apply]
    #update the background model
    # fgMask = backSub.apply(frame)
    fgMask = cv.subtract(frame, init_frame)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(fgMask,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ## [apply]

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    # cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    # cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
            #    cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    ## [show]
    #show the current frame and the fg masks
    cv.imshow('Frame', frame)
    cv.imshow('FG Mask', th3)
    ## [show]

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
