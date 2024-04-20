# TensorRT Acceleration for YoloV9 Applications

This repository provides a comprehensive framework for enhancing YoloV9 object detection applications with NVIDIA TensorRT. The goal is to optimize YoloV9 models to achieve faster inference times and improved performance on NVIDIA GPUs, making real-time object detection more efficient and scalable.

## Features

- **TensorRT Optimization**: Implementations to convert YoloV9 models to TensorRT optimized engines that significantly reduce latency and increase throughput on compatible hardware.
- **Example Applications**: Sample applications demonstrating the use of optimized YoloV9 models in real-world scenarios.
- **Comprehensive Documentation**: Detailed instructions for converting models, setting up environments, and deploying optimized detectors.

## Getting Started

### Installation
For installation instructions, see [Installation Guide](INSTALL.md).

### TensorRT inference

* Inference (Todo: Create a standalone detect.py)
```
cd yolov9
python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights './yolov9-c.engine' --name yolov9_c_c_640_detect
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
