cmake_minimum_required(VERSION 3.10)
project(VideoProcessor VERSION 1.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set OpenCV_CUDA_VERSION to avoid the error
set(OpenCV_STATIC OFF)
set(OpenCV_CUDA_VERSION "")

# Find OpenCV package with additional DNN module
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio cudaarithm cudaoptflow optflow cudawarping cudaimgproc dnn dnn_superres)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Optional CUDA support - using modern approach
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_definitions(-DWITH_CUDA)
    message(STATUS "CUDA found, enabling GPU acceleration")
else()
    message(STATUS "CUDA not found, falling back to CPU implementation")
endif()

# Create models directory if it doesn't exist
file(MAKE_DIRECTORY ${PROJECT_SOURCE_DIR}/models)

# Common source files (updated with new enhancement modules)
set(COMMON_SOURCES
    src/camera.cpp
    src/timer.cpp
    src/frame_buffer.cpp
    src/upscaler.cpp
    src/dnn_super_res.cpp
    # New enhancement modules
    src/temporal_consistency.cpp
    src/adaptive_sharpening.cpp
    src/selective_bilateral.cpp
)

# Phase 4 additional sources
set(PHASE4_SOURCES
    src/display.cpp
    src/processor.cpp
    src/pipeline.cpp
)

# Add main executable with all sources
add_executable(video_processor 
    src/main.cpp
    ${COMMON_SOURCES}
    ${PHASE4_SOURCES}
)

# Link libraries to main executable
target_link_libraries(video_processor ${OpenCV_LIBS})

# Add simple OpenCV test
add_executable(opencv_test src/opencv_test.cpp)
target_link_libraries(opencv_test ${OpenCV_LIBS})

# Add simple camera test
add_executable(simple_camera_test src/simple_camera_test.cpp)
target_link_libraries(simple_camera_test ${OpenCV_LIBS})

# Add Phase 2 test
add_executable(test_phase2
    src/test_phase2.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(test_phase2 ${OpenCV_LIBS})

# Add Phase 4 test
add_executable(test_phase4
    src/test_phase4.cpp
    ${COMMON_SOURCES}
    ${PHASE4_SOURCES}
)
target_link_libraries(test_phase4 ${OpenCV_LIBS})

# Add enhancements test application
add_executable(test_enhancements
    src/test_enhancements.cpp
    ${COMMON_SOURCES}
)
target_link_libraries(test_enhancements ${OpenCV_LIBS})

# Link CUDA if available
if(CMAKE_CUDA_COMPILER)
    # For CUDA-specific libraries if needed
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        target_link_libraries(video_processor CUDA::cudart)
        target_link_libraries(test_phase2 CUDA::cudart)
        target_link_libraries(test_phase4 CUDA::cudart)
        target_link_libraries(test_enhancements CUDA::cudart)
        target_link_libraries(opencv_test CUDA::cudart)
        
        # Set CUDA source properties for the new enhancement modules
        set_source_files_properties(
            src/temporal_consistency.cpp
            src/adaptive_sharpening.cpp
            src/selective_bilateral.cpp
            PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ
        )
    endif()
endif()

# Provide compile commands for tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output binary to bin directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Copy model files to bin directory
file(COPY ${PROJECT_SOURCE_DIR}/models DESTINATION ${EXECUTABLE_OUTPUT_PATH})

# Add a message to remind about downloading models
message(STATUS "")
message(STATUS "NOTE: Remember to download ML models to the 'models' directory.")
message(STATUS "You can download models from: https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres/models")
message(STATUS "Example: wget -P models https://github.com/opencv/opencv_contrib/raw/master/modules/dnn_superres/models/FSRCNN_x4.pb")
message(STATUS "")
message(STATUS "NOTE: For RealESRGAN, you need to download the ONNX model:")
message(STATUS "Example: wget -P models https://github.com/XPixelGroup/BasicSR/releases/download/v1.4.0/RRDB_ESRGAN_x4.pth")
message(STATUS "Then convert it to ONNX using the provided convert_realesrgan.py script:")
message(STATUS "python convert_realesrgan.py models/RRDB_ESRGAN_x4.pth models/RRDB_ESRGAN_x4.onnx")
message(STATUS "")

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "  CUDA Support: ${CMAKE_CUDA_COMPILER}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Output Directory: ${EXECUTABLE_OUTPUT_PATH}")
message(STATUS "")
message(STATUS "Targets:")
message(STATUS "  video_processor - Main application")
message(STATUS "  test_phase2 - Phase 2 test application")
message(STATUS "  test_phase4 - Phase 4 test application")
message(STATUS "  test_enhancements - New enhancement modules test application")
message(STATUS "  opencv_test - OpenCV capabilities test")
message(STATUS "  simple_camera_test - Basic camera functionality test")
message(STATUS "")
message(STATUS "New Enhancement Modules:")
message(STATUS "  Temporal Consistency - Reduces flickering and improves stability")
message(STATUS "  Adaptive Sharpening - Intelligently sharpens details while preserving smooth areas")
message(STATUS "  Selective Bilateral - Edge-preserving noise reduction and detail preservation")
message(STATUS "")