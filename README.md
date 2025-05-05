Low-Latency Video Processing System
A high-performance video processing system built with C++ and OpenCV, designed for low-latency video capture, processing, and display.
Project Overview
This project provides a framework for capturing video streams from cameras, processing frames with various algorithms including upscaling, and displaying the results with minimal latency. The system is designed with a multi-threaded architecture that separates capturing, processing, and display operations for maximum performance.
Key features:

Camera capture with automatic camera detection
Thread-safe zero-copy frame buffer for efficient producer-consumer pattern
GPU-accelerated frame processing (when CUDA is available)
Performance timing and statistics
Upscaling algorithms with quality/performance options
Background super-resolution processing for maintaining quality without sacrificing frame rate
Adaptive frame skipping for smooth playback under load
Memory optimization to reduce allocations
Input downscaling for large frames to improve processing efficiency

Directory Structure
├── .vscode/                # VSCode configuration
├── bin/                    # Compiled executables 
├── include/                # Header files
│   ├── camera.h            # Camera capture interface
│   ├── display.h           # Display interface
│   ├── dnn_super_res.h     # DNN-based super-resolution
│   ├── frame_buffer.h      # Thread-safe frame buffer
│   ├── pipeline.h          # Processing pipeline coordinator
│   ├── processor.h         # Frame processor interface
│   ├── timer.h             # Performance timing utility
│   └── upscaler.h          # Frame upscaling with CPU/GPU implementations
├── src/                    # Source files
│   ├── camera.cpp          # Camera implementation
│   ├── display.cpp         # Display implementation
│   ├── dnn_super_res.cpp   # DNN-based super-resolution implementation
│   ├── frame_buffer.cpp    # Frame buffer implementation
│   ├── main.cpp            # Main application entry point
│   ├── pipeline.cpp        # Pipeline implementation
│   ├── processor.cpp       # Processor implementation
│   ├── timer.cpp           # Timer implementation
│   └── upscaler.cpp        # Upscaler implementation (CPU & GPU)
├── test/                   # Test files
└── CMakeLists.txt          # CMake build configuration
Performance Optimizations
The system includes several optimizations to maximize performance while maintaining high-quality output:

Background Super-Resolution: Performs computationally intensive super-resolution in a separate thread to avoid blocking the main processing pipeline.
Adaptive Frame Skipping: Intelligently skips frames when the buffer is filling up to maintain smooth playback.
Input Downscaling: Automatically downscales large input frames before processing to reduce computational load.
FP16 Precision: Uses half-precision floating point when available on CUDA devices to accelerate neural network inference.
Memory Optimization: Reuses existing memory allocations when possible to reduce memory overhead.
Quality/Performance Trade-offs: Implements dynamic quality adjustments based on system load.

Building the Project
Prerequisites:

CMake 3.10 or higher
C++17 compatible compiler
OpenCV 4.x with DNN and optionally CUDA modules
CUDA Toolkit (for GPU acceleration)

Build steps:
bash# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build
make

# Run the main application
../bin/video_processor
Command Line Options
The application supports several command line options for customizing behavior:

<path> - Path to video file or camera index (default: 0)
--output <path> or -o <path> - Path for saving processed video
--record or -r - Start recording immediately
--fast-mode or -fm - Use faster processing algorithms (lower quality)

Testing
Several test programs are provided:

simple_camera_test: Basic camera functionality test
opencv_test: OpenCV environment test
test_phase2: Tests the frame buffer and upscaler components
test_phase4: Tests the complete integrated pipeline

GPU Acceleration
The system automatically detects if CUDA is available and enables GPU acceleration for compatible operations. The upscaler can use GPU acceleration for faster performance when processing high-resolution frames.
Performance Tips

For optimal performance with super-resolution, ensure you have a CUDA-capable GPU.
Use the --fast-mode option when processing very high-resolution videos.
For better quality without sacrificing frame rate, the default mode now uses background super-resolution processing.
Execute test_phase4 for benchmarking the full pipeline.
When recording, ensure your storage device has sufficient speed to handle the output data rate.

Notes:

The adaptive processing system balances quality and performance automatically.
You can press 'q' to quit, 'r' to toggle recording, and 's' to take a snapshot during playback.
If frames are still being dropped despite optimizations, consider further reducing input resolution or using a more powerful GPU.
