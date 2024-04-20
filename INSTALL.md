# Installation

## Prerequisites

- NVIDIA GPU with support for CUDA
- CUDA Toolkit (v12.1)
- cuDNN
- Python 3.9, 3.10, or 3.11 (recommend create a conda environment)

## Install TensorRT 

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

## Install yolov9

- Install yolov9 github repo.
```bash
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9
```

## Model Conversion. 
` Use yolov9's export method to get onnx model and then use TensoRT tool to convert onnx to engine model.
```bash
python yolov9/export.py --weights yolov9-c-converted.pt --include onnx
trtexec --onnx=yolov9-c-converted.onnx --explicitBatch --saveEngine=yolov9-c.engine --fp16
```

