#include "camera.h"
#include "frame_buffer.h"
#include "upscaler.h"
#include "timer.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>
#include <opencv2/opencv.hpp>

// Flag to signal threads to exit
std::atomic<bool> g_running(true);

// Producer thread function
void producer_thread(Camera& camera, FrameBuffer& buffer, Timer& timer) {
    std::cout << "Producer thread started" << std::endl;
    cv::Mat frame;
    int frame_count = 0;
    
    while (g_running) {
        // Measure frame acquisition time
        timer.start("acquisition");
        bool success = camera.getFrame(frame);
        timer.stop("acquisition");
        
        if (!success || frame.empty()) {
            std::cerr << "Failed to get frame from camera" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Push frame to buffer, measure time
        timer.start("buffer_push");
        bool pushed = buffer.pushFrame(frame, false); // Non-blocking
        timer.stop("buffer_push");
        
        if (pushed) {
            frame_count++;
            if (frame_count % 100 == 0) {
                std::cout << "Produced " << frame_count << " frames" << std::endl;
            }
        } else {
            std::cerr << "Buffer full, frame dropped" << std::endl;
        }
    }
    
    std::cout << "Producer thread finished after " << frame_count << " frames" << std::endl;
}

// Consumer thread function
void consumer_thread(FrameBuffer& buffer, Upscaler& upscaler, Timer& timer) {
    std::cout << "Consumer thread started" << std::endl;
    cv::Mat frame, upscaled_frame;
    int frame_count = 0;
    
    // Create window for display
    cv::namedWindow("Upscaled Feed", cv::WINDOW_NORMAL);
    
    while (g_running) {
        // Get frame from buffer, measure time
        timer.start("buffer_pop");
        bool success = buffer.popFrame(frame, false); // Non-blocking
        timer.stop("buffer_pop");
        
        if (!success) {
            // No frame available, wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Upscale the frame, measure time
        timer.start("upscale");
        bool upscale_success = upscaler.upscale(frame, upscaled_frame);
        timer.stop("upscale");
        
        if (!upscale_success) {
            std::cerr << "Failed to upscale frame" << std::endl;
            continue;
        }
        
        // Calculate and display FPS on the frame
        frame_count++;
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(1000.0 / 
                              (timer.getAverageDuration("buffer_pop") + 
                               timer.getAverageDuration("upscale"))));
        
        std::string res_text = "Resolution: " + std::to_string(upscaled_frame.cols) + 
                              "x" + std::to_string(upscaled_frame.rows);
        
        cv::putText(upscaled_frame, fps_text, cv::Point(20, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(upscaled_frame, res_text, cv::Point(20, 60), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        
        // Display upscaled frame
        cv::imshow("Upscaled Feed", upscaled_frame);
        
        // Print stats every 100 frames
        if (frame_count % 100 == 0) {
            std::cout << "Consumed " << frame_count << " frames" << std::endl;
            timer.printStats();
        }
        
        // Check for quit
        if (cv::waitKey(1) == 'q') {
            g_running = false;
        }
    }
    
    cv::destroyWindow("Upscaled Feed");
    std::cout << "Consumer thread finished after " << frame_count << " frames" << std::endl;
}

int main() {
    std::cout << "Phase 2 Test: Zero-Copy Buffer and Upscaler" << std::endl;
    
    // Check for available cameras
    auto cameras = Camera::listAvailableCameras();
    if (cameras.empty()) {
        std::cerr << "No cameras detected!" << std::endl;
        return -1;
    }
    
    // Initialize camera
    Camera camera(cameras[0]);
    if (!camera.initialize()) {
        std::cerr << "Failed to initialize camera" << std::endl;
        return -1;
    }
    
    // Get camera properties
    int src_width = camera.getWidth();
    int src_height = camera.getHeight();
    std::cout << "Camera resolution: " << src_width << "x" << src_height << std::endl;
    
    // Target resolution (1080p)
    int target_width = 1920;
    int target_height = 1080;
    
    // Create frame buffer with capacity of 10 frames
    FrameBuffer buffer(10);
    
    // Create upscaler
    Upscaler upscaler(Upscaler::BILINEAR, true);
    if (!upscaler.initialize(target_width, target_height)) {
        std::cerr << "Failed to initialize upscaler" << std::endl;
        return -1;
    }
    
    std::cout << "Using " << (upscaler.isUsingGPU() ? "GPU" : "CPU") 
              << " upscaling with " << upscaler.getAlgorithmName() << std::endl;
    
    // Create timer
    Timer timer;
    
    // Start producer and consumer threads
    std::cout << "Starting threads..." << std::endl;
    std::thread producer(producer_thread, std::ref(camera), std::ref(buffer), std::ref(timer));
    std::thread consumer(consumer_thread, std::ref(buffer), std::ref(upscaler), std::ref(timer));
    
    // Wait for user to press Enter to exit
    std::cout << "Press Enter to stop..." << std::endl;
    std::cin.get();
    
    // Signal threads to exit and wait for them
    g_running = false;
    producer.join();
    consumer.join();
    
    std::cout << "Final statistics:" << std::endl;
    timer.printStats();
    
    return 0;
}