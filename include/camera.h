#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <memory>
#include <vector>

class Camera {
public:
    // Constructor with camera index or video file
    Camera(int camera_index = 0);
    Camera(const std::string& video_source);
    ~Camera();

    // Initialize the camera with specific resolution and framerate
    bool initialize(int width = 1280, int height = 720, int fps = 60);
    
    // Get the next frame (non-blocking)
    bool getFrame(cv::Mat& frame);
    
    // Check if camera is opened successfully
    bool isOpened() const;
    
    // Get camera properties
    double getFPS() const;
    int getWidth() const;
    int getHeight() const;
    
    // List available cameras on the system
    static std::vector<int> listAvailableCameras();
    
    // Try different camera backends
    bool tryBackends();
    
private:
    std::unique_ptr<cv::VideoCapture> cap;
    int width;
    int height;
    int fps;
    bool initialized;
    int camera_index;
    std::string video_source;
    bool is_file;
};