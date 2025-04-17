#include "camera.h"
#include "frame_buffer.h"
#include "upscaler.h"
#include "timer.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <filesystem>
#include <signal.h>

// Global control flag
std::atomic<bool> g_running(true);

// Global statistics
std::atomic<int> g_frames_captured(0);
std::atomic<int> g_frames_processed(0);
std::atomic<int> g_frames_displayed(0);
std::atomic<int> g_frames_dropped(0);
std::atomic<bool> g_save_video(false);

// Global video writer, moved outside so it can be properly closed on exit
std::unique_ptr<cv::VideoWriter> g_video_writer;
bool g_writer_initialized = false;
std::string g_output_filename = "output.mp4";

// Signal handler for clean shutdown
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    g_running = false;
    
    // Give the threads a chance to exit cleanly
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // If video was being recorded, make sure it's closed properly
    if (g_writer_initialized && g_video_writer) {
        std::cout << "Closing video file..." << std::endl;
        g_video_writer->release();
        std::cout << "Video saved to: " << g_output_filename << std::endl;
    }
    
    exit(signum);
}

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
            std::cerr << "Failed to get frame from source" << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            
            // Check if we've reached the end of a video file
            if (!camera.isOpened()) {
                std::cout << "End of video file reached" << std::endl;
                g_running = false;
                break;
            }
            
            continue;
        }
        
        // Calculate actual camera FPS
        frame_counter++;
        auto current_time = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
        
        if (elapsed >= 1.0) {
            fps = frame_counter / elapsed;
            std::cout << "Source capture rate: " << fps << " FPS" << std::endl;
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
        
        std::string recording_text = g_save_video ? "RECORDING" : "";
        
        // Add text with lightweight method
        cv::putText(processed_frame, fps_text, cv::Point(20, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(processed_frame, buffer_text, cv::Point(20, 60), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(processed_frame, dropped_text, cv::Point(20, 90), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                    
        // Show recording status if we're saving video
        if (g_save_video) {
            cv::putText(processed_frame, recording_text, cv::Point(processed_frame.cols - 200, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        }
        
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
void display_thread(FrameBuffer& buffer, Timer& timer, 
                    double fps, int width, int height) {
    std::cout << "Display thread started" << std::endl;
    cv::Mat frame;
    
    // Create window with a consistent size
    cv::namedWindow("Video Feed", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Feed", 1280, 720);

    // VideoWriter parameters
    int codec = cv::VideoWriter::fourcc('X', '2', '6', '4'); // H.264 codec
    double output_fps = (fps > 0) ? fps : 30.0; // Use source fps or default to 30
    
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
        
        // Initialize video writer when the first valid frame is available and saving is enabled
        if (g_save_video && !g_writer_initialized && !frame.empty()) {
            try {
                // Create output directory if it doesn't exist
                std::filesystem::path output_path = g_output_filename;
                std::filesystem::create_directories(output_path.parent_path());
                
                // Make sure we use an absolute path
                if (!output_path.is_absolute()) {
                    output_path = std::filesystem::absolute(output_path);
                    g_output_filename = output_path.string();
                }
                
                std::cout << "Creating video file: " << g_output_filename << std::endl;
                
                // Initialize video writer with the frame's properties
                g_video_writer = std::make_unique<cv::VideoWriter>(
                    g_output_filename, 
                    codec, 
                    output_fps, 
                    cv::Size(frame.cols, frame.rows)
                );
                
                if (g_video_writer->isOpened()) {
                    g_writer_initialized = true;
                    std::cout << "Video recording started: " << g_output_filename << std::endl;
                    std::cout << "Output resolution: " << frame.cols << "x" << frame.rows << std::endl;
                    std::cout << "Output FPS: " << output_fps << std::endl;
                } else {
                    std::cerr << "Failed to create video writer" << std::endl;
                    g_save_video = false;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error creating video writer: " << e.what() << std::endl;
                g_save_video = false;
            }
        }
        
        // Write the frame to video file if saving is enabled
        if (g_save_video && g_writer_initialized && !frame.empty()) {
            timer.start("video_write");
            g_video_writer->write(frame);
            timer.stop("video_write");
        }
        
        // Display frame with minimal processing
        timer.start("display_show");
        cv::imshow("Video Feed", frame);
        timer.stop("display_show");
        
        g_frames_displayed++;
        
        // Check for key commands
        int key = cv::waitKey(1);
        if (key == 'q') {
            g_running = false;
            break;
        } else if (key == 'r') {
            // Toggle recording
            g_save_video = !g_save_video;
            
            if (g_save_video) {
                if (!g_writer_initialized) {
                    std::cout << "Video recording will start with the next frame" << std::endl;
                } else {
                    std::cout << "Video recording resumed" << std::endl;
                }
            } else {
                std::cout << "Video recording paused" << std::endl;
            }
        } else if (key == 's') {
            // Save a single frame as an image
            std::string snapshot_filename = "snapshot_" + 
                                            std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + 
                                            ".jpg";
            cv::imwrite(snapshot_filename, frame);
            std::cout << "Snapshot saved to " << snapshot_filename << std::endl;
        }
    }
    
    // Clean up the video writer
    if (g_writer_initialized && g_video_writer) {
        g_video_writer->release();
        std::cout << "Video recording finished and saved to: " << g_output_filename << std::endl;
    }
    
    cv::destroyAllWindows();
    std::cout << "Display thread finished" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "Low-Latency Video Processing System" << std::endl;
    
    // Register signal handler for clean shutdown
    signal(SIGINT, signalHandler);  // Ctrl+C
    signal(SIGTERM, signalHandler); // termination request
    
    // Check if a video file path is provided as a command-line argument
    std::string video_source;
    bool use_video_file = false;
    
    // Process command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                g_output_filename = argv[++i];
                std::cout << "Output will be saved to: " << g_output_filename << std::endl;
            }
        } else if (arg == "--record" || arg == "-r") {
            g_save_video = true;
            std::cout << "Recording will start automatically" << std::endl;
        } else {
            // Assume this is the input source (file or camera index)
            video_source = arg;
            
            // Check if it's a number (camera index) or a string (file path)
            bool is_number = true;
            for (size_t j = 0; j < video_source.length(); j++) {
                if (!std::isdigit(video_source[j])) {
                    is_number = false;
                    break;
                }
            }
            
            if (is_number) {
                // It's a camera index
                std::cout << "Using camera index: " << video_source << std::endl;
                use_video_file = false;
            } else {
                // It's a file path
                std::cout << "Using video file: " << video_source << std::endl;
                use_video_file = true;
            }
        }
    }
    
    // Create camera or video source
    std::unique_ptr<Camera> source;
    if (use_video_file) {
        source = std::make_unique<Camera>(video_source);
    } else {
        // List available cameras
        auto available_cameras = Camera::listAvailableCameras();
        
        if (available_cameras.empty()) {
            std::cerr << "No cameras detected! Please connect a camera or provide a video file path." << std::endl;
            std::cerr << "Usage: " << argv[0] << " [camera_index|video_file_path] [--output filename] [--record]" << std::endl;
            return -1;
        }
        
        // Use the first available camera or the specified one
        int camera_id = available_cameras[0];
        if (!video_source.empty()) {
            camera_id = std::stoi(video_source);
            // Validate camera index
            if (std::find(available_cameras.begin(), available_cameras.end(), camera_id) == available_cameras.end()) {
                std::cout << "Camera index " << camera_id << " not available." << std::endl;
                std::cout << "Using camera index " << available_cameras[0] << " instead." << std::endl;
                camera_id = available_cameras[0];
            }
        }
        source = std::make_unique<Camera>(camera_id);
    }
    
    // Initialize camera or video source
    if (!source->initialize(640, 480, 60)) {
        std::cerr << "Error: Could not initialize with preferred settings" << std::endl;
        std::cerr << "Trying with default settings..." << std::endl;
        
        if (!source->initialize()) {
            std::cerr << "Error: Could not initialize with default settings" << std::endl;
            return -1;
        }
    }
    
    double source_fps = source->getFPS();
    int source_width = source->getWidth();
    int source_height = source->getHeight();
    
    std::cout << "Source initialized successfully at " 
              << source_width << "x" << source_height
              << " @ " << source_fps << " FPS" << std::endl;
    
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
    
    std::thread capture(capture_thread, std::ref(*source), std::ref(raw_buffer), std::ref(timer));
    std::thread processor(processing_thread, std::ref(raw_buffer), std::ref(processed_buffer), 
                         std::ref(upscaler), std::ref(timer));
    std::thread display(display_thread, std::ref(processed_buffer), std::ref(timer), 
                        source_fps, source_width, source_height);
    
    std::cout << "Pipeline running. Press 'q' in the video window to quit." << std::endl;
    std::cout << "Press 'r' to toggle recording, 's' to take a snapshot." << std::endl;
    
    // Wait for threads to be done
    capture.join();
    processor.join();
    display.join();
    
    // Ensure video writer is properly closed
    if (g_writer_initialized && g_video_writer) {
        g_video_writer->release();
        std::cout << "Video saved to: " << g_output_filename << std::endl;
    }
    
    // Print final statistics
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total frames captured:  " << g_frames_captured << std::endl;
    std::cout << "Total frames processed: " << g_frames_processed << std::endl;
    std::cout << "Total frames displayed: " << g_frames_displayed << std::endl;
    std::cout << "Total frames dropped:   " << g_frames_dropped << std::endl;
    
    timer.printStats();
    
    return 0;
}