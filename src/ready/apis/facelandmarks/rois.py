"""
python rois.py
https://pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/
https://github.com/PyImageSearch/imutils/blob/master/imutils/face_utils/helpers.py
"""
import os
import time
import numpy as np
import argparse
import dlib
import cv2
from collections import OrderedDict
from pathlib import Path
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_file", help="Config filename with path", type=str)
args = vars(parser.parse_args())
config_file = args["config_file"]
config = OmegaConf.load(config_file)

DATA_PATH = config.datapath
PREDICTOR = config.shape_predictor
FULL_DATA_PATH = os.path.join(Path.home(), DATA_PATH)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])


# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(FULL_DATA_PATH+"/"+PREDICTOR)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
# time.sleep(2.0)


# frame_i=0
new_width, new_height = 600, 480
threshold=0

while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    ret, frame = vs.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections
    for rect in rects:
        # # determine the facial landmarks for the face region, then
        # # convert the facial landmark (x, y)-coordinates to a NumPy
        # # array
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        # print(f"shape {shape}")


        for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
            clone = frame.copy()
            # print(f"index:{i}")
            # print(f"name: {name}")
            # grab the (x, y)-coordinates associated with the face landmark
            (i, j) = FACIAL_LANDMARKS_IDXS[name]
            if name == "right_eye":
                for (x,y) in shape[i:j]:
                    cv2.circle(clone, (x,y), 1, (0,0,255),-1)
                # print(f"np.array([shape: {np.array([shape[i:j]])}")
                (x_rect, y_rect, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                # (x_cir,y_cir),radius = cv2.minEnclosingCircle(np.array([shape[i:j]]))
                # print(x_rect, int(x_cir), y_rect, int(y_cir))
                roi0 = frame[y_rect:y_rect + h + threshold, x_rect:x_rect + w + threshold]
                roi0 = cv2.resize(roi0, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("ROI_rigtheye", roi0)
    
            if name == "left_eye":
                for (x,y) in shape[i:j]:
                    cv2.circle(clone, (x,y), 1, (0,0,255),-1)
                (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
                roi1 = frame[y:y + h, x:x + w]
                roi1 = cv2.resize(roi1, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                cv2.imshow("ROI_left_eye", roi1)


        # # loop over the (x, y)-coordinates for the facial landmarks
        # # and draw them on the image
        for (x, y) in shape:
            # print(f"x y {x, y}")
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
            cv2.putText(frame, "Face landmarks", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # print(f"frame: {frame_i}")
    # frame_i = frame_i + 1
    # show the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("q pressed")
        break

