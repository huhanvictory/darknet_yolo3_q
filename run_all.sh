#!/bin/bash

IMAGE_PATH="/Users/reza/Downloads/Dataset/VOC/test/VOCdevkit2/VOC2007/JPEGImages/*.jpg"
WEIGHT_PATH="/Users/reza/Github/falcon_yolo_quantizer/model/gen/Yolov3_q.weights"
CONFIG_PATH="/Users/reza/Github/falcon_yolo_quantizer/model/gen/Yolov3_q.cfg"

for f in $IMAGE_PATH
do
  ./darknet detect $CONFIG_PATH $WEIGHT_PATH $f -thresh 0.2
done

