#include "camera.h"
#include "timer.h"
#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "Low-Latency Video Processing System" << std::endl;
    
    // Create camera with default device
    Camera camera(0);
    
    // Initialize with native settings
    if (!camera.initialize()) {
        std::cerr << "Error: Could not initialize camera" << std::endl;
        return -1;
    }
    
    // Create timer
    Timer timer;
    
    // Main capture loop
    cv::Mat frame;
    int frame_count = 0;
    
    std::cout << "Starting capture loop. Press 'q' to quit." << std::endl;
    
    while (true) {
        // Capture frame
        timer.start("frame_acquisition");
        bool success = camera.getFrame(frame);
        timer.stop("frame_acquisition");
        
        if (!success || frame.empty()) {
            std::cerr << "Failed to capture frame" << std::endl;
            break;
        }
        
        // Show frame info
        frame_count++;
        std::string res_info = std::to_string(frame.cols) + "x" + std::to_string(frame.rows);
        cv::putText(frame, "Frame: " + std::to_string(frame_count), cv::Point(20, 30), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Resolution: " + res_info, cv::Point(20, 60), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Display the frame
        cv::imshow("Camera Feed", frame);
        
        // Print stats every 100 frames
        if (frame_count % 100 == 0) {
            timer.printStats();
        }
        
        // Check for quit
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
    
    // Clean up
    cv::destroyAllWindows();
    
    std::cout << "Capture complete. Processed " << frame_count << " frames." << std::endl;
    timer.printStats();
    
    return 0;
}