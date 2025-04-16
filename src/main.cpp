#include "camera.h"
#include "frame_buffer.h"
#include "upscaler.h"
#include "timer.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <chrono>

// Global control flag
std::atomic<bool> g_running(true);

// Global statistics
std::atomic<int> g_frames_captured(0);
std::atomic<int> g_frames_processed(0);
std::atomic<int> g_frames_displayed(0);
std::atomic<int> g_frames_dropped(0);

// Capture thread function - optimized to prevent buffer overflow
void capture_thread(Camera& camera, FrameBuffer& buffer, Timer& timer) {
    std::cout << "Capture thread started" << std::endl;
    cv::Mat frame;
    
    // For measuring real capture rate
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_counter = 0;
    double fps = 0;
    
    while (g_running) {
        // Time the frame acquisition
        timer.start("acquisition");
        bool success = camera.getFrame(frame);
        timer.stop("acquisition");
        
        if (!success || frame.empty()) {
            std::cerr << "Failed to get frame from camera" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        
        // Calculate actual camera FPS
        frame_counter++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
        
        if (elapsed >= 1.0) {
            fps = frame_counter / elapsed;
            std::cout << "Camera capture rate: " << fps << " FPS" << std::endl;
            frame_counter = 0;
            start_time = current_time;
        }
        
        // Adaptive frame dropping - only push if buffer isn't almost full
        if (buffer.size() < buffer.capacity() * 0.8) {
            // Push frame to buffer (non-blocking)
            timer.start("buffer_push");
            bool pushed = buffer.pushFrame(frame, false);
            timer.stop("buffer_push");
            
            if (pushed) {
                g_frames_captured++;
            } else {
                g_frames_dropped++;
                if (g_frames_dropped % 10 == 0) {
                    std::cout << "Warning: Dropped " << g_frames_dropped << " frames due to full buffer" << std::endl;
                }
            }
        } else {
            // Skip frame due to buffer pressure
            g_frames_dropped++;
            if (g_frames_dropped % 10 == 0) {
                std::cout << "Warning: Dropped " << g_frames_dropped << " frames due to full buffer" << std::endl;
            }
            
            // Add short delay to allow consumer to catch up
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    std::cout << "Capture thread finished" << std::endl;
}

// Processing thread function - separate from display
void processing_thread(FrameBuffer& input_buffer, FrameBuffer& output_buffer, 
                        Upscaler& upscaler, Timer& timer) {
    std::cout << "Processing thread started" << std::endl;
    cv::Mat input_frame, processed_frame;
    
    while (g_running) {
        // Get frame from input buffer
        timer.start("buffer_pop");
        bool success = input_buffer.popFrame(input_frame, false);
        timer.stop("buffer_pop");
        
        if (!success) {
            // No frame available, yield time slice
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Process the frame - time each major operation separately
        timer.start("upscale");
        bool upscale_success = upscaler.upscale(input_frame, processed_frame);
        timer.stop("upscale");
        
        if (!upscale_success) {
            std::cerr << "Failed to upscale frame" << std::endl;
            continue;
        }
        
        // Add performance metrics text to the frame
        timer.start("text_overlay");
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(1000.0 / 
                            (timer.getAverageDuration("upscale"))));
        
        std::string buffer_text = "Buffer: " + std::to_string(input_buffer.size()) + 
                                "/" + std::to_string(input_buffer.capacity());
        
        std::string dropped_text = "Dropped: " + std::to_string(g_frames_dropped);
        
        // Add text with lightweight method
        cv::putText(processed_frame, fps_text, cv::Point(20, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(processed_frame, buffer_text, cv::Point(20, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(processed_frame, dropped_text, cv::Point(20, 90), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        timer.stop("text_overlay");
        
        // Push to output buffer
        timer.start("output_push");
        output_buffer.pushFrame(processed_frame, false);
        timer.stop("output_push");
        
        g_frames_processed++;
        
        // Print detailed stats periodically
        if (g_frames_processed % 100 == 0) {
            std::cout << "\nProcessed " << g_frames_processed << " frames" << std::endl;
            timer.printStats();
        }
    }
    
    std::cout << "Processing thread finished" << std::endl;
}

// Display thread function - minimal work, just showing frames
void display_thread(FrameBuffer& buffer, Timer& timer) {
    std::cout << "Display thread started" << std::endl;
    cv::Mat frame;
    
    // Create window with a consistent size
    cv::namedWindow("Video Feed", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Feed", 1280, 720);
    
    while (g_running) {
        // Get processed frame
        timer.start("display_pop");
        bool success = buffer.popFrame(frame, false);
        timer.stop("display_pop");
        
        if (!success) {
            // No frame available, wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Display frame with minimal processing
        timer.start("display_show");
        cv::imshow("Video Feed", frame);
        timer.stop("display_show");
        
        g_frames_displayed++;
        
        // Check for quit command - use shorter wait time
        if (cv::waitKey(1) == 'q') {
            g_running = false;
            break;
        }
    }
    
    cv::destroyAllWindows();
    std::cout << "Display thread finished" << std::endl;
}

int main() {
    std::cout << "Low-Latency Video Processing System" << std::endl;
    
    // List available cameras
    auto available_cameras = Camera::listAvailableCameras();
    
    if (available_cameras.empty()) {
        std::cerr << "No cameras detected! Please connect a camera and try again." << std::endl;
        return -1;
    }
    
    // Use the first available camera
    int camera_id = available_cameras[0];
    
    // Create camera with selected device
    Camera camera(camera_id);
    
    // Initialize camera with 720p (or lower if not supported)
    if (!camera.initialize(1280, 720, 30)) {
        std::cerr << "Error: Could not initialize camera with preferred settings" << std::endl;
        std::cerr << "Trying with default settings..." << std::endl;
        
        if (!camera.initialize()) {
            std::cerr << "Error: Could not initialize camera with default settings" << std::endl;
            return -1;
        }
    }
    
    std::cout << "Camera initialized successfully at " 
              << camera.getWidth() << "x" << camera.getHeight()
              << " @ " << camera.getFPS() << " FPS" << std::endl;
    
    // Create upscaler with target Full HD resolution
    // Using bilinear for better performance
    Upscaler upscaler(Upscaler::BILINEAR, true);
    if (!upscaler.initialize(1920, 1080)) {
        std::cerr << "Error: Could not initialize upscaler" << std::endl;
        return -1;
    }
    
    std::cout << "Upscaler initialized with algorithm: " 
              << upscaler.getAlgorithmName()
              << ", using " << (upscaler.isUsingGPU() ? "GPU" : "CPU") << std::endl;
    
    // Create frame buffers with increased capacity
    // Buffer between capture and processing threads
    FrameBuffer raw_buffer(20);
    // Buffer between processing and display threads
    FrameBuffer processed_buffer(10);
    
    std::cout << "Frame buffers initialized with sizes 20 and 10" << std::endl;
    
    // Create timer for performance measurement
    Timer timer;
    
    // Start all threads
    std::cout << "Starting pipeline threads..." << std::endl;
    
    std::thread capture(capture_thread, std::ref(camera), std::ref(raw_buffer), std::ref(timer));
    std::thread processor(processing_thread, std::ref(raw_buffer), std::ref(processed_buffer), 
                         std::ref(upscaler), std::ref(timer));
    std::thread display(display_thread, std::ref(processed_buffer), std::ref(timer));
    
    std::cout << "Pipeline running. Press 'q' in the video window to quit." << std::endl;
    
    // Wait for threads to be done
    capture.join();
    processor.join();
    display.join();
    
    // Print final statistics
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total frames captured:  " << g_frames_captured << std::endl;
    std::cout << "Total frames processed: " << g_frames_processed << std::endl;
    std::cout << "Total frames displayed: " << g_frames_displayed << std::endl;
    std::cout << "Total frames dropped:   " << g_frames_dropped << std::endl;
    
    timer.printStats();
    
    return 0;
}