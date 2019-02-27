import argparse
from os import path
import time
import logging
import sys
import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
from object_detector_detection_api import ObjectDetectorDetectionAPI


basepath = path.dirname(__file__)

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt=' %I:%M:%S ',
    level="INFO"
)
logger = logging.getLogger('detector')

if __name__ == '__main__':

    # initialize detector
    logger.info('Model loading...')
    predictor = ObjectDetectorDetectionAPI(path.join(basepath, "frozen_inference_graph.pb"))


    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 32
    rawCapture = PiRGBArray(camera, size=(640, 480))

    # allow the camera to warmup
    time.sleep(0.1)

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    count =0 
    # capture frames from the camera
    for frame in camera.capture_continuous(rawCapture, format="bgr",
                                           use_video_port=True):
        t1 = cv2.getTickCount()
        if count >=10:
            logger.info("warning person missing")
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        image = frame.array

        logger.info("FPS: {0:.2f}".format(frame_rate_calc))
        cv2.putText(image, "FPS: {0:.2f}".format(frame_rate_calc), (20, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2, cv2.LINE_AA)

        result = predictor.detect(image)
        foundPerson = False
        for obj in result:
            logger.info('coordinates: {} {}. class: "{}". confidence: {:.2f}'.
                        format(obj[0], obj[1], obj[3], obj[2]))
            
            if obj[3] == "person":
                foundPerson = True
            cv2.rectangle(image, obj[0], obj[1], (0, 255, 0), 2)
            cv2.putText(image, '{}: {:.2f}'.format(obj[3], obj[2]),
                        (obj[0][0], obj[0][1] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

        if not foundPerson:
            count=count+1
        else:
            count = 0
        # show the frame
        cv2.imshow("Stream", image)
        key = cv2.waitKey(1) & 0xFF

        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1

        # clear the stream in preparation for the next frame
        rawCapture.truncate(0)
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break