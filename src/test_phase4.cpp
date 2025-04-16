#include "pipeline.h"
#include "camera.h"  // Add direct include for Camera class
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

int main(int argc, char* argv[]) {
    std::cout << "Phase 4 Test: Integrated Pipeline with Display and Latency Optimization" << std::endl;
    
    // Create pipeline configuration
    Pipeline::Config config;
    
    // Parse command-line arguments for camera index
    if (argc > 1) {
        config.camera_index = std::stoi(argv[1]);
    }
    
    // Check for available cameras
    auto cameras = Camera::listAvailableCameras();
    if (cameras.empty()) {
        std::cerr << "No cameras detected!" << std::endl;
        return -1;
    }
    
    // Make sure the camera index is valid
    if (std::find(cameras.begin(), cameras.end(), config.camera_index) == cameras.end()) {
        std::cout << "Camera index " << config.camera_index << " not available." << std::endl;
        config.camera_index = cameras[0];
        std::cout << "Using camera index " << config.camera_index << " instead." << std::endl;
    }
    
    // Configure pipeline for testing
    config.camera_width = 1280;
    config.camera_height = 720;
    config.target_width = 1920;
    config.target_height = 1080;
    config.buffer_size = 3;  // Smaller buffer for lower latency
    config.upscale_algorithm = Upscaler::BILINEAR;
    config.use_gpu = true;
    config.show_metrics = true;
    config.window_name = "Phase 4 Test";
    
    // Create and initialize pipeline
    Pipeline pipeline(config);
    if (!pipeline.initialize()) {
        std::cerr << "Failed to initialize pipeline" << std::endl;
        return -1;
    }
    
    // Start the pipeline
    if (!pipeline.start()) {
        std::cerr << "Failed to start pipeline" << std::endl;
        return -1;
    }
    
    std::cout << "Pipeline started successfully" << std::endl;
    std::cout << "Press 'q' to quit, 'p' to print stats" << std::endl;
    
    // Main loop
    bool running = true;
    auto start_time = std::chrono::steady_clock::now();
    
    while (running) {
        // Check for key presses
        int key = cv::waitKey(1);
        
        if (key == 'q' || key == 27) {  // 'q' or ESC
            running = false;
        }
        else if (key == 'p') {
            pipeline.printPerformanceStats();
        }
        
        // Print stats periodically
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();
        
        if (elapsed >= 5) {
            std::cout << "\n=== Pipeline Status ===" << std::endl;
            std::cout << "Latency: " << std::fixed << std::setprecision(2) 
                      << pipeline.getLatency() << " ms" << std::endl;
            std::cout << "FPS: " << std::fixed << std::setprecision(1) 
                      << pipeline.getFPS() << std::endl;
            
            start_time = now;
        }
        
        // Small sleep to avoid consuming 100% CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Stop the pipeline
    std::cout << "Stopping pipeline..." << std::endl;
    pipeline.stop();
    
    // Print final stats
    std::cout << "\n=== Final Performance Statistics ===" << std::endl;
    pipeline.printPerformanceStats();
    
    return 0;
}