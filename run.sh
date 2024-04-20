#!/bin/bash

python yolov9/export.py --weights yolov9-c-converted.pt --include onnx


trtexec --onnx=yolov9-c-converted.onnx --explicitBatch --saveEngine=yolov9-c.engine --fp16


python yolov9/detect.py --source horses.jpg --img 640 --device 0 --weights yolov9-c.engine --name yolov9_c_c_640_detect

cd yolov9
python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c.engine' --name yolov9_c_c_640_detect