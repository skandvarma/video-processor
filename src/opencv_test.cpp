#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h> // For access() function

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // Check available backends by trying to open them with a non-existent device
    std::cout << "Testing backends:" << std::endl;
    
    std::vector<std::pair<int, std::string>> backends = {
        {cv::CAP_ANY, "AUTO"},
        {cv::CAP_V4L, "V4L"},
        {cv::CAP_V4L2, "V4L2"},
        {cv::CAP_GSTREAMER, "GStreamer"},
        {cv::CAP_FFMPEG, "FFMPEG"}
    };
    
    for (const auto& [backend, name] : backends) {
        // Directly test camera with this backend
        cv::VideoCapture cap(0, backend);
        bool works = cap.isOpened();
        std::cout << name << ": " << (works ? "Working" : "Not working") << std::endl;
        
        if (works) {
            // Try to get properties
            int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            double fps = cap.get(cv::CAP_PROP_FPS);
            
            std::cout << "  - Resolution: " << width << "x" << height << std::endl;
            std::cout << "  - FPS: " << fps << std::endl;
            
            // Test grabbing a frame
            cv::Mat frame;
            bool grabbed = cap.read(frame);
            std::cout << "  - Frame grab: " << (grabbed ? "Success" : "Failed") << std::endl;
            if (grabbed) {
                std::cout << "  - Frame size: " << frame.cols << "x" << frame.rows << std::endl;
            }
            
            // Release the camera for the next test
            cap.release();
        }
    }
    
    // Check video device files
    std::cout << "\nChecking video devices:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::string device = "/dev/video" + std::to_string(i);
        if (access(device.c_str(), F_OK) != -1) {
            std::cout << device << " exists" << std::endl;
            
            // Check if readable
            if (access(device.c_str(), R_OK) != -1) {
                std::cout << "  - Readable: Yes" << std::endl;
            } else {
                std::cout << "  - Readable: No (permission issue)" << std::endl;
            }
        }
    }
    
    // Try direct V4L2 commands to check device
    std::cout << "\nAttempting to check /dev/video0 directly:" << std::endl;
    FILE* fp = popen("v4l2-ctl --device=/dev/video0 --all 2>&1", "r");
    if (fp) {
        char buffer[128];
        while (fgets(buffer, sizeof(buffer), fp) != NULL) {
            std::cout << buffer;
        }
        pclose(fp);
    }
    
    return 0;
}