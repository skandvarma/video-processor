#include "camera.h"
#include "timer.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <thread>

int main(int argc, char* argv[]) {
    std::cout << "Low-Latency Video Processing System" << std::endl;
    
    // List available cameras
    std::cout << "Checking available cameras..." << std::endl;
    auto available_cameras = Camera::listAvailableCameras();
    
    if (available_cameras.empty()) {
        std::cerr << "No cameras detected! Please connect a camera and try again." << std::endl;
        return -1;
    }
    
    // Use the first available camera or user-specified camera
    int camera_id = available_cameras[0];
    if (argc > 1) {
        camera_id = std::stoi(argv[1]);
    }
    
    std::cout << "Using camera: " << camera_id << std::endl;
    
    // Create camera with selected device
    Camera camera(camera_id);
    
    // Initialize with native resolution
    if (!camera.initialize()) {
        std::cerr << "Error: Could not initialize camera" << std::endl;
        return -1;
    }
    
    // Create timer for performance measurement
    Timer timer;
    
    // Variables for the main loop
    cv::Mat frame;
    int frame_count = 0;
    
    // For FPS calculation
    auto fps_start_time = std::chrono::high_resolution_clock::now();
    int fps_counter = 0;
    double actual_fps = 0.0;
    
    std::cout << "Starting capture loop. Press 'q' to quit." << std::endl;
    
    while (true) {
        // Measure frame acquisition time
        timer.start("frame_acquisition");
        bool success = camera.getFrame(frame);
        timer.stop("frame_acquisition");
        
        if (!success || frame.empty()) {
            std::cerr << "Error: Failed to capture frame" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Update FPS calculation
        fps_counter++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - fps_start_time).count();
        
        if (elapsed >= 1.0) {
            actual_fps = fps_counter / elapsed;
            fps_counter = 0;
            fps_start_time = current_time;
        }
        
        // Display information on the frame
        frame_count++;
        std::string camera_fps = "Camera FPS: " + std::to_string(camera.getFPS());
        std::string actual_fps_str = "Actual FPS: " + std::to_string(actual_fps);
        std::string res_info = "Resolution: " + std::to_string(frame.cols) + "x" + std::to_string(frame.rows);
        std::string frame_info = "Frame: " + std::to_string(frame_count);
        
        cv::putText(frame, camera_fps, cv::Point(20, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, actual_fps_str, cv::Point(20, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, res_info, cv::Point(20, 90), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, frame_info, cv::Point(20, 120), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Add timing information
        double acq_time = timer.getDuration("frame_acquisition");
        std::string timing = "Acquisition: " + std::to_string(acq_time) + " ms";
        cv::putText(frame, timing, cv::Point(20, 150), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Display the frame
        cv::imshow("Camera Feed", frame);
        
        // Print timing stats every 100 frames
        if (frame_count % 100 == 0) {
            timer.printStats();
        }
        
        // Check for quit command
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