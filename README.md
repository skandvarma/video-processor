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

## Directory Structure

```
├── .vscode/                # VSCode configuration
├── bin/                    # Compiled executables 
├── include/                # Header files
│   ├── camera.h            # Camera capture interface
│   ├── display.h           # Display interface (placeholder)
│   ├── frame_buffer.h      # Thread-safe frame buffer
│   ├── processor.h         # Frame processor interface (placeholder)
│   ├── timer.h             # Performance timing utility
│   └── upscaler.h          # Frame upscaling with CPU/GPU implementations
├── src/                    # Source files
│   ├── camera.cpp          # Camera implementation
│   ├── display.cpp         # Display implementation (placeholder)
│   ├── frame_buffer.cpp    # Frame buffer implementation
│   ├── main.cpp            # Main application entry point
│   ├── opencv_test.cpp     # OpenCV environment test utility
│   ├── processor.cpp       # Processor implementation (placeholder)
│   ├── simple_camera_test.cpp # Simple camera test utility
│   ├── test_phase2.cpp     # Phase 2 testing (buffer + upscaler)
│   ├── timer.cpp           # Timer implementation
│   └── upscaler.cpp        # Upscaler implementation (CPU & GPU)
├── test/                   # Test files (placeholder)
└── CMakeLists.txt          # CMake build configuration
```

## File Descriptions

### Configuration Files

- **.vscode/c_cpp_properties.json**: VSCode C/C++ extension configuration file that sets up include paths for OpenCV.

- **CMakeLists.txt**: CMake build system configuration that handles:
  - Setting C++17 standard
  - Finding and linking OpenCV dependencies
  - Optional CUDA detection and linking
  - Building multiple executables for testing and production
  - Setting output paths

### Core Components

#### Camera Module

- **include/camera.h**: Defines the `Camera` class interface for video capture, supporting:
  - Camera index and video file sources
  - Resolution and framerate configuration
  - Camera detection and initialization
  - Frame retrieval

- **src/camera.cpp**: Implementation of the Camera class that abstracts OpenCV's VideoCapture functionality with:
  - Support for detecting available cameras
  - Automatic retrieval of camera properties
  - Robust initialization and error handling

#### Frame Buffer

- **include/frame_buffer.h**: Defines a thread-safe ring buffer for passing frames between threads:
  - Uses atomic operations and mutexes for thread safety
  - Implements blocking and non-blocking operations
  - Zero-copy design using OpenCV's reference counting
  - Condition variables for producer/consumer coordination

- **src/frame_buffer.cpp**: Implementation of the FrameBuffer class with:
  - Efficient frame management
  - Thread synchronization
  - Overflow handling

#### Upscaler

- **include/upscaler.h**: Interface for frame upscaling with multiple algorithms:
  - Multiple interpolation methods (Nearest, Bilinear, Bicubic, Lanczos)
  - Super-resolution support
  - Automatic GPU/CPU selection

- **src/upscaler.cpp**: Implementation with:
  - CPU implementation using OpenCV's resize functions
  - Optional GPU implementation using CUDA when available
  - Algorithm selection and switching
  - Pimpl pattern for implementation details

#### Timer

- **include/timer.h**: Performance measurement utility for profiling:
  - Named timing events
  - Statistics collection (min, max, average)
  - Multiple simultaneous timers

- **src/timer.cpp**: Implementation with:
  - High-resolution clock usage
  - Statistical aggregation
  - Formatted output

### Application and Tests

- **src/main.cpp**: Main application that:
  - Detects and initializes a camera
  - Captures and displays video frames
  - Measures and displays performance metrics
  - Provides an interactive UI

- **src/test_phase2.cpp**: Test program for Phase 2 features:
  - Separate producer and consumer threads
  - Frame buffer for thread communication
  - Upscaling processing step
  - Performance measurement

- **src/opencv_test.cpp**: Diagnostic utility to:
  - Check OpenCV version
  - Test available camera backends
  - Verify device access
  - Test V4L2 configuration

- **src/simple_camera_test.cpp**: Minimal camera test that:
  - Initializes a camera
  - Captures a single frame
  - Reports camera properties

### Placeholder Files

These files are empty or minimal placeholders for future implementation:

- **include/display.h** and **src/display.cpp**: For dedicated display functionality
- **include/processor.h** and **src/processor.cpp**: For additional frame processing
- **test/** directory files: For future unit testing

## Building the Project

Prerequisites:
- CMake 3.10 or higher
- C++17 compatible compiler
- OpenCV 4.x
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

## Testing

Several test programs are provided:

- **simple_camera_test**: Basic camera functionality test
- **opencv_test**: OpenCV environment test
- **test_phase2**: Tests the frame buffer and upscaler components

## GPU Acceleration

The system automatically detects if CUDA is available and enables GPU acceleration for compatible operations. The upscaler can use GPU acceleration for faster performance when processing high-resolution frames.

## Performance Considerations

- The zero-copy frame buffer minimizes memory usage and copying overhead
- Thread separation allows parallel execution of capture and processing
- The timer component helps identify performance bottlenecks
- GPU acceleration provides significant speedup for compute-intensive operations

## Future Enhancements

- Full implementation of the display module
- Additional processing algorithms (edge detection, motion tracking, etc.)
- More comprehensive testing suite
- Configuration file support
- Recording functionality
