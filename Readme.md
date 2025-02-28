# YOLOv11 C++ Implementation

A high-performance C++ implementation of YOLOv11 object detection using ONNX Runtime and OpenCV.

![YOLO Detection](https://raw.githubusercontent.com/ultralytics/assets/main/yolov11/banner-yolov11.png)

## Features

- Fast and efficient object detection using YOLOv11
- Support for both CPU and GPU inference (CUDA)
- Video processing capabilities
- Dynamic confidence and IoU thresholds
- Visual performance metrics (FPS counter)
- Semi-transparent bounding box masks for cleaner visualization

## Prerequisites

- CMake 3.12+
- C++17 compatible compiler
- OpenCV 4.x
- ONNX Runtime 1.17+
- CUDA Toolkit (optional, for GPU acceleration)

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/yolov11cpp.git
cd yolov11cpp
```

### Building with CMake

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Prepare the Model

1. Export your YOLOv11 model to ONNX format using Ultralytics:

```bash
# If using Python/Ultralytics
yolo export model=yolov11s.pt format=onnx opset=12 simplify=True
```

2. Place your ONNX model and class names file in the project directory:

```bash
cp path/to/best.onnx ./
cp path/to/classes.txt ./
```

## Usage

### Basic Command

```bash
./yolov11_detector [options]
```

### Options

- `--model`: Path to the ONNX model file (default: "./best.onnx")
- `--classes`: Path to the class names file (default: "./classes.txt")
- `--input`: Path to input video file or camera device index (default: "./input.mov")
- `--output`: Path for output video file (default: "./output.mp4")
- `--gpu`: Use GPU acceleration if available (default: false)
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for NMS (default: 0.45)

### Example

```bash
# Process a video file with custom thresholds
./yolov11_detector --input=test_video.mp4 --output=result.mp4 --conf=0.3 --iou=0.4

# Use webcam (device 0) with GPU acceleration
./yolov11_detector --input=0 --gpu=true
```

## Configuration

You can modify the default settings by editing the constants in:

- `src/camera_inference.cpp` - Main application settings
- `src/ia/YOLO11.hpp` - Detection parameters and algorithms
- `src/ia/tools/Config.hpp` - Debug and timing configurations

## Debugging

Enable debugging by uncommenting these lines in `src/ia/tools/Config.hpp`:

```cpp
// Enable debug messages
#define DEBUG_MODE

// Enable performance timing
#define TIMING_MODE
```

## Troubleshooting

### Accuracy Issues

If you notice differences in detection accuracy compared to the Python implementation:

1. Verify your ONNX model is exported correctly with proper settings
2. Check that preprocessing matches Ultralytics implementation (RGB conversion, normalization)
3. Confirm your class names file is correct and in the expected format
4. Try adjusting the confidence and IoU thresholds to match Ultralytics defaults (0.25 and 0.45)

### Performance Issues

- For CPU optimization, ensure `ORT_ENABLE_ALL` optimization is enabled
- For GPU usage, verify CUDA toolkit and ONNX Runtime with CUDA support are installed
- Reduce input image resolution for better performance

[Take Reference](https://github.com/Geekgineer/YOLOs-CPP)