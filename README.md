# Low-Latency Video Processing System

A high-performance video processing system built with C++ and OpenCV, designed for low-latency video capture, processing, and display.

## Project Overview

This project provides a framework for capturing video streams from cameras, processing frames with various algorithms including upscaling, and displaying the results with minimal latency. The system is designed with a multi-threaded architecture that separates capturing, processing, and display operations for maximum performance.

Key features:
- Camera capture with automatic camera detection
- Thread-safe zero-copy frame buffer for efficient producer-consumer pattern
- GPU-accelerated frame processing (when CUDA is available)
- Performance timing and statistics
- Upscaling algorithms with quality/performance options
- Background Super-Resolution processing for high quality without compromising FPS
- Adaptive frame skipping to maintain smooth playback
- Automatic downscaling for large input frames

## Directory Structure
```
├── .vscode/                # VSCode configuration
├── bin/                    # Compiled executables
├── include/                # Header files
│   ├── camera.h            # Camera capture interface
│   ├── display.h           # Display interface
│   ├── dnn_super_res.h     # Deep learning super-resolution
│   ├── frame_buffer.h      # Thread-safe frame buffer
│   ├── pipeline.h          # Processing pipeline coordinator
│   ├── processor.h         # Frame processor interface
│   ├── timer.h             # Performance timing utility
│   ├── upscaler.h          # Frame upscaling with CPU/GPU implementations
│   └── video_enhancer.h    # Video enhancement effects
├── models/                 # ML models for super-resolution
├── src/                    # Source files
│   ├── camera.cpp          # Camera implementation
│   ├── display.cpp         # Display implementation
│   ├── dnn_super_res.cpp   # DNN-based super-resolution implementation
│   ├── frame_buffer.cpp    # Frame buffer implementation
│   ├── main.cpp            # Main application entry point
│   ├── opencv_test.cpp     # OpenCV environment test utility
│   ├── pipeline.cpp        # Pipeline implementation
│   ├── processor.cpp       # Processor implementation
│   ├── simple_camera_test.cpp # Simple camera test utility
│   ├── test_phase2.cpp     # Phase 2 testing (buffer + upscaler)
│   ├── test_phase4.cpp     # Phase 4 testing (full pipeline)
│   ├── timer.cpp           # Timer implementation
│   ├── upscaler.cpp        # Upscaler implementation
│   └── video_enhancer.cpp  # Video enhancement implementation
├── test/                   # Test files
└── CMakeLists.txt          # CMake build configuration
```
## Performance Optimizations

The system includes several key performance optimizations:

1. **Background Super-Resolution Processing**: Super-resolution is performed in a separate thread to avoid blocking the main pipeline, allowing smooth video playback while still benefiting from high-quality upscaling.

2. **Adaptive Frame Skipping**: The system dynamically adjusts frame skipping based on buffer fullness to prevent overflow and maintain smooth playback.

3. **Input Downscaling**: Very large input frames are automatically downscaled before processing to reduce computational load.

4. **FP16 Precision for DNNs**: Uses half-precision floating-point (FP16) for neural network inference on compatible GPUs, significantly improving performance.

5. **Memory Optimization**: Reuses existing frame buffers to minimize memory allocations and reduce garbage collection overhead.

6. **Optimized GPU/CPU Switching**: Automatically selects the best processing method based on hardware availability and performance requirements.

## Building the Project

Prerequisites:
- CMake 3.10 or higher
- C++17 compatible compiler
- OpenCV 4.x with DNN and optionally CUDA modules
- Optional: CUDA for GPU acceleration

Build steps:
```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
make

# Run the main application
../bin/video_processor
```
Command Line Options
The application supports several command-line options:
```
./bin/video_processor [options] [source]

source          - Camera index (number) or video file path
--output, -o    - Specify output file for recording
--record, -r    - Start recording immediately
--fast-mode, -fm - Use faster algorithms for better performance
--help, -h      - Show help information

# Process a video file with super-resolution and save the output
./bin/video_processor my_video.mp4 --output enhanced_video.mp4
```

Performance Tips

For maximum performance, run on a system with a CUDA-capable GPU
Use lower resolution input sources for faster processing
Try the --fast-mode option when highest quality isn't required
Use test_phase2 or test_phase4 for optimized testing
Adjust buffer sizes in the code for your specific hardware

Troubleshooting

If you experience frame drops, your system might not be fast enough for super-resolution. Try the --fast-mode option.
If no cameras are detected, ensure you have proper permissions for camera access
Check CUDA availability with the opencv_test utility
Monitor system resource usage during processing to identify bottlenecks
