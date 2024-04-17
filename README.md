# TensorRT Acceleration for YoloV9 Applications

This repository provides a comprehensive framework for enhancing YoloV9 object detection applications with NVIDIA TensorRT. The goal is to optimize YoloV9 models to achieve faster inference times and improved performance on NVIDIA GPUs, making real-time object detection more efficient and scalable.

## Features

- **TensorRT Optimization**: Implementations to convert YoloV9 models to TensorRT optimized engines that significantly reduce latency and increase throughput on compatible hardware.
- **Example Applications**: Sample applications demonstrating the use of optimized YoloV9 models in real-world scenarios.
- **Comprehensive Documentation**: Detailed instructions for converting models, setting up environments, and deploying optimized detectors.

## Getting Started

### Prerequisites

- NVIDIA GPU with support for CUDA
- CUDA Toolkit (v12.1)
- cuDNN
- Python 3.9, 3.10, or 3.11 (recommend create a conda environment)

- Steps to install TensorRT 8.6.1

```bash
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-12.0.tar.gz
cd TensorRT-8.6.1.6
cd python
pip install tensorrt-8.6.1-cp310-none-linux_x86_64.whl
pip install tensorrt_lean-8.6.1-cp310-none-linux_x86_64.whl
pip install tensorrt_dispatch-8.6.1-cp310-none-linux_x86_64.whl
cd ../onnx_graphsurgeon/
pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl 
```

Add lines in the end of '~/.bashrc` (change the TensorRT path accordingly.)
```bash
export LD_LIBRARY_PATH=/home/zhipeng/tensorrt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
export PATH=/home/zhipeng/tensorrt/TensorRT-8.6.1.6/bin${PATH:+:${PATH}}
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tensorrt-yolov9.git
   cd tensorrt-yolov9
   ```

2. I use https://github.com/laugh12321/TensorRT-YOLO (v2.0) to perform model converting and inference:
   ```bash
   git clone https://github.com/laugh12321/TensorRT-YOLO  # clone
   cd TensorRT-YOLO
   pip install -r requirements.txt 
   pip install ultralytics      
   ```

### TensorRT inference

* Model converting
```
python TensorRT-YOLO/python/export/yolov9/export.py -w yolov9-c-converted.pt -o output -b 8 --img 640 -s
trtexec --onnx=yolov9-c-converted.onnx --saveEngine=yolov9-c-converted.engine --fp16
```

* Inference
```
python TensorRT-YOLO/detect.py -e yolov9-c-converted.engine -o output -i horses.jpg -l TensorRT-YOLO/labels.txt
```



## To do
- [ ] Dynamic TensorRT model converting.
- [ ] Objective tracking.
- [ ] Speed estimation and counting.
- [ ] Object Segmentation (Segment-Anything)

## Contributing

Contributions are welcome! If you'd like to improve the TensorRT optimizations for YoloV9, add new features, or provide examples, please feel free to fork the repository and submit a pull request.

## License

This project is licensed under **GPL-3.0**, as found in the LICENSE file.
