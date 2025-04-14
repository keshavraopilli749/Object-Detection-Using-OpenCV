import cv2
import numpy as np
import os

# Thresholds
thres = 0.45
nms_threshold = 0.5

# Webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)

# Load class names
classFile = os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names')
with open(os.path.realpath(classFile), 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model
configPath = os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
weightsPath = os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb')

net = cv2.dnn_DetectionModel(os.path.realpath(weightsPath), os.path.realpath(configPath))
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, image = cap.read()
    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():  # Use flatten instead of i[0]
            box = bbox[i]
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            classId = int(classIds[i]) - 1  # Get the class ID and convert to 0-based index
            cv2.putText(image, classNames[classId], (x + 10, y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", image)
    cv2.waitKey(1)
