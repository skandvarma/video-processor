#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera" << std::endl;
        return -1;
    }
    
    cv::Mat frame;
    cap >> frame;
    
    if (frame.empty()) {
        std::cerr << "Failed to capture frame" << std::endl;
    } else {
        std::cout << "Successfully captured frame: " 
                 << frame.cols << "x" << frame.rows << std::endl;
        
        // Show some information about the camera
        std::cout << "Camera properties:" << std::endl;
        std::cout << "Resolution: " << cap.get(cv::CAP_PROP_FRAME_WIDTH) 
                 << "x" << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
        std::cout << "FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;
    }
    
    return 0;
}