// camera.cpp (simplified version)
#include "camera.h"
#include <iostream>
#include <thread>

Camera::Camera(int camera_index) 
    : width(0), height(0), fps(0), initialized(false), camera_index(camera_index), is_file(false) {
}

Camera::Camera(const std::string& video_source) 
    : width(0), height(0), fps(0), initialized(false), camera_index(-1), 
      video_source(video_source), is_file(true) {
}

Camera::~Camera() {
    if (cap && cap->isOpened()) {
        cap->release();
    }
}

std::vector<int> Camera::listAvailableCameras() {
    std::vector<int> availableCameras;
    
    for (int i = 0; i < 10; i++) {
        cv::VideoCapture temp(i);
        if (temp.isOpened()) {
            std::cout << "Camera " << i << " is available" << std::endl;
            availableCameras.push_back(i);
            temp.release();
        }
    }
    
    return availableCameras;
}

bool Camera::initialize(int w, int h, int framerate) {
    // Open the camera directly with the simplest approach
    cap = std::make_unique<cv::VideoCapture>(camera_index);
    
    if (!cap->isOpened()) {
        std::cerr << "Failed to open camera with default backend" << std::endl;
        return false;
    }
    
    std::cout << "Camera opened successfully" << std::endl;
    
    // Get actual camera capabilities
    width = static_cast<int>(cap->get(cv::CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(cap->get(cv::CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<int>(cap->get(cv::CAP_PROP_FPS));
    
    std::cout << "Camera native resolution: " 
              << width << "x" << height 
              << " @ " << fps << " FPS" << std::endl;
    
    // Don't try to set properties that may not be supported
    // Just use whatever we get natively
    
    // Verify we can get a frame
    cv::Mat test_frame;
    bool success = cap->read(test_frame);
    
    if (!success || test_frame.empty()) {
        std::cerr << "Failed to read test frame" << std::endl;
        return false;
    }
    
    std::cout << "Successfully captured test frame: " 
              << test_frame.cols << "x" << test_frame.rows << std::endl;
    
    initialized = true;
    return true;
}

bool Camera::getFrame(cv::Mat& frame) {
    if (!cap || !cap->isOpened() || !initialized) {
        return false;
    }
    
    return cap->read(frame);
}

bool Camera::isOpened() const {
    return (cap != nullptr);
}

double Camera::getFPS() const {
    return fps;
}

int Camera::getWidth() const {
    return width;
}

int Camera::getHeight() const {
    return height;
}