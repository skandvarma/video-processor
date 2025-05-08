#include "camera.h"
#include "frame_buffer.h"
#include "upscaler.h"
#include "temporal_consistency.h"
#include "adaptive_sharpening.h"
#include "selective_bilateral.h"
#include "timer.h"
#include <iostream>
#include <thread>
#include <atomic>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <filesystem>
#include <signal.h>
#include <opencv2/video/tracking.hpp>


// Global control flag
std::atomic<bool> g_running(true);

// Global statistics
std::atomic<int> g_frames_captured(0);
std::atomic<int> g_frames_processed(0);
std::atomic<int> g_frames_displayed(0);
std::atomic<int> g_frames_dropped(0);
std::atomic<bool> g_save_video(false);
std::queue<cv::Mat> g_sr_frame_queue;
std::mutex g_sr_queue_mutex;
std::atomic<bool> g_sr_processing_active(false);
cv::Mat g_last_sr_result;
std::mutex g_sr_result_mutex;
cv::Mat g_previous_frame;
float g_blend_alpha = 0.0f;
std::mutex g_blend_mutex;
std::deque<cv::Mat> g_frame_history;
const int HISTORY_SIZE = 3;
std::mutex g_history_mutex;
bool g_using_super_res = false;
std::string g_output_format = "mp4";

// Global video writer, moved outside so it can be properly closed on exit
std::unique_ptr<cv::VideoWriter> g_video_writer;
bool g_writer_initialized = false;
std::string g_output_filename = "output.mp4";

// Add this function for temporal smoothing without optical flow
cv::Mat createSmoothFrame(const std::deque<cv::Mat>& frame_history) {
    if (frame_history.size() < 2) {
        return frame_history.back().clone();
    }
    
    // Get latest frame
    cv::Mat current = frame_history.back();
    
    // Create result frame
    cv::Mat result = current.clone();
    
    // Apply temporal blending with decreasing weights
    float total_weight = 0.0f;
    float weights[HISTORY_SIZE] = {0.7f, 0.2f, 0.1f}; // Most weight on current frame
    
    for (int i = 0; i < frame_history.size() && i < HISTORY_SIZE; i++) {
        int idx = frame_history.size() - 1 - i;
        float weight = weights[i];
        total_weight += weight;
        
        if (i == 0) {
            // First frame (latest) - initialize result
            cv::addWeighted(result, 0, frame_history[idx], weight, 0, result);
        } else {
            // Add weighted previous frames
            cv::addWeighted(result, 1.0, frame_history[idx], weight, 0, result);
        }
    }
    
    // Normalize if needed
    if (total_weight != 1.0f && total_weight > 0.0f) {
        result = result * (1.0f / total_weight);
    }
    
    return result;
}

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

// Capture thread function - with frame rate control for video files
void capture_thread(Camera& camera, FrameBuffer& buffer, Timer& timer, 
                   bool is_video_file, double target_fps, bool using_super_res) {
    std::cout << "Capture thread started" << std::endl;
    cv::Mat frame;

    // For measuring real capture rate
    auto start_time = std::chrono::high_resolution_clock::now();
    int frame_counter = 0;
    double fps = 0;

    // For frame rate control
    auto last_frame_time = std::chrono::high_resolution_clock::now();
    std::chrono::microseconds frame_interval(0);

    // Calculate frame interval if we have a valid FPS
    if (is_video_file && target_fps > 0) {
        frame_interval = std::chrono::microseconds(static_cast<long long>(1000000.0 / target_fps));
        std::cout << "Video frame rate control enabled: Target " << target_fps 
                  << " FPS (interval: " << frame_interval.count() << "µs)" << std::endl;
    }

    // For controlled frame skipping
    int frame_skip = using_super_res ? (is_video_file ? 1 : 1) : 1;
    int frame_skip_counter = 0;

    while (g_running) {
        // For video files, control frame rate to match source
        if (is_video_file && frame_interval.count() > 0) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = now - last_frame_time;

            if (elapsed < frame_interval) {
                // Wait until it's time for the next frame
                std::this_thread::sleep_for(frame_interval - elapsed);
            }
            last_frame_time = std::chrono::high_resolution_clock::now();
        }

        // Controlled frame skipping for super-res
        frame_skip_counter++;
        if (using_super_res && frame_skip_counter % frame_skip != 0) {
            // For super-res, skip frames at regular intervals to allow processing to keep up
            // This is different from buffer pressure-based skipping

            // We still need to read the frame to advance the video
            if (is_video_file) {
                cv::Mat temp_frame;
                camera.getFrame(temp_frame);
            }

            continue;
        }

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

        // Instead of pure buffer pressure-based frame dropping, use a more
        // sophisticated approach that considers both buffer state and processing method

        // Calculate buffer utilization
        double buffer_utilization = static_cast<double>(buffer.size()) / buffer.capacity();

        if (buffer_utilization < 0.9) {  // Only push if buffer isn't almost full
            // Push frame to buffer - use blocking for more consistent behavior
            timer.start("buffer_push");
            bool pushed = buffer.pushFrame(frame, true);
            timer.stop("buffer_push");

            if (pushed) {
                g_frames_captured++;
            } else {
                g_frames_dropped++;
                std::cerr << "Failed to push frame to buffer" << std::endl;
            }
        } else {
            // Buffer is too full, we must skip this frame
            g_frames_dropped++;

            if (g_frames_dropped % 10 == 0) {
                std::cout << "Warning: Dropped " << g_frames_dropped << " frames due to full buffer" << std::endl;
            }

            // When buffer is almost full, add a longer delay to allow consumer to catch up
            std::this_thread::sleep_for(std::chrono::milliseconds(10 * using_super_res ? 30 : 10));
        }
    }

    std::cout << "Capture thread finished" << std::endl;
}

void processing_thread(FrameBuffer& input_buffer, FrameBuffer& output_buffer, 
                          Upscaler& upscaler, Timer& timer) {
    std::cout << "Processing thread started" << std::endl;
    cv::Mat input_frame, processed_frame;

    // For tracking performance
    double avg_processing_time = 0.0;

    // FIXED: Properly check if using super-resolution based on algorithm
    bool g_using_super_res = (upscaler.getAlgorithmName() == "RealESRGAN" || 
                             upscaler.getAlgorithmName() == "Standard Super-Res");
    
    std::cout << "Processing with " << (g_using_super_res ? "Super-Resolution" : "Bicubic") 
             << " algorithm" << std::endl;

    while (g_running) {
        // Get frame from input buffer - use blocking mode to avoid busy waiting
        timer.start("buffer_pop");
        bool success = input_buffer.popFrame(input_frame, true);
        timer.stop("buffer_pop");

        if (!success || input_frame.empty()) {
            // This should rarely happen with blocking mode, but just in case
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Reduce input resolution if it's too high, especially for super-res
        if (g_using_super_res && (input_frame.cols > 480 || input_frame.rows > 270)) {
            cv::Mat resized_input;
            double scale = std::min(480.0 / input_frame.cols, 270.0 / input_frame.rows);
            cv::resize(input_frame, resized_input, cv::Size(), scale, scale, cv::INTER_AREA);
            input_frame = resized_input;
        }

        // Process the frame with the upscaler
        timer.start("upscale");
        bool upscale_success = upscaler.upscale(input_frame, processed_frame);
        timer.stop("upscale");

        if (!upscale_success || processed_frame.empty()) {
            std::cerr << "Upscaling failed, using original input" << std::endl;
            // If upscaling fails, resize the input to target size as fallback
            cv::resize(input_frame, processed_frame, cv::Size(upscaler.getTargetWidth(), upscaler.getTargetHeight()), 
                      0, 0, cv::INTER_CUBIC);
        }

        // Apply temporal blending for smoother transitions
        {
            std::lock_guard<std::mutex> lock(g_history_mutex);

            // Add current frame to history
            if (g_frame_history.size() >= HISTORY_SIZE) {
                g_frame_history.pop_front();
            }
            g_frame_history.push_back(processed_frame.clone());

            // Create a smooth transition by blending frames
            if (g_frame_history.size() >= 2) {
                // Get weights for blending based on whether we're using SR
                float current_weight = g_using_super_res ? 0.7f : 0.6f;
                float prev_weight = 1.0f - current_weight;

                // Blend current frame with previous frame
                cv::addWeighted(
                  g_frame_history.back(), current_weight, 
                  g_frame_history[g_frame_history.size() - 2], prev_weight, 
                  0, processed_frame
                );
            }
        }

        // Track processing time with exponential moving average
        double current_processing_time = timer.getDuration("upscale");
        avg_processing_time = (avg_processing_time * 0.9) + (current_processing_time * 0.1);

        // Add performance metrics text
        timer.start("text_overlay");
        std::string fps_text = "FPS: " + std::to_string(static_cast<int>(1000.0 / 
                  (avg_processing_time + 
                   timer.getAverageDuration("buffer_pop") + 
                   timer.getAverageDuration("output_push"))));

        std::string buffer_text = "Buffer: " + std::to_string(input_buffer.size()) + 
                      "/" + std::to_string(input_buffer.capacity());

        std::string proc_text = "Process: " + std::to_string(static_cast<int>(avg_processing_time)) + " ms";

        std::string mode_text = "Mode: " + (g_using_super_res ? upscaler.getAlgorithmName() : "Bicubic") + 
                     " + Temporal Smoothing";

        // Add text overlay - green for bicubic, orange for super-res
        cv::Scalar text_color = g_using_super_res ? cv::Scalar(0, 165, 255) : cv::Scalar(0, 255, 0);

        cv::putText(processed_frame, fps_text, cv::Point(20, 30), 
          cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
        cv::putText(processed_frame, buffer_text, cv::Point(20, 60), 
          cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
        cv::putText(processed_frame, proc_text, cv::Point(20, 90), 
          cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);
        cv::putText(processed_frame, mode_text, cv::Point(20, 120), 
          cv::FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2);

        timer.stop("text_overlay");

        // Push to output buffer - use non-blocking and handle potential failure
        timer.start("output_push");
        // Check output buffer fullness before pushing
        if (output_buffer.size() >= output_buffer.capacity() * 0.9) {
            // Wait briefly to allow display thread to consume frames
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        bool pushed = output_buffer.pushFrame(processed_frame, false);
        timer.stop("output_push");

        if (pushed) {
            g_frames_processed++;
        } else {
            // If push failed, wait for display thread to catch up
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Print detailed stats periodically
        if (g_frames_processed % 100 == 0) {
            std::cout << "\nProcessed " << g_frames_processed << " frames" << std::endl;
            std::cout << "Current processing time: " << avg_processing_time << " ms" << std::endl;
            std::cout << "Buffer utilization: " << input_buffer.size() << "/" << input_buffer.capacity() << std::endl;
            timer.printStats();
        }
    }

    std::cout << "Processing thread finished" << std::endl;
}

// Add this display thread function to your main.cpp file just before the main() function

void displayLoop(FrameBuffer& buffer, Timer& timer,
                 double fps, int width, int height) {
    std::cout << "Display thread started" << std::endl;
    cv::Mat frame;
    
    // Create window with a consistent size
    cv::namedWindow("Video Feed", cv::WINDOW_NORMAL);
    cv::resizeWindow("Video Feed", 640, 480);

    // Default video parameters
    int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // Default MP4V codec
    std::string extension = ".mp4";      // Default extension
    double output_fps = fps;             // Use source FPS to maintain correct timing
    
    // Configure codec and extension based on format
    if (g_output_format == "mp4") {
        codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        extension = ".mp4";
        std::cout << "Using MP4 format with MP4V codec" << std::endl;
    }
    else if (g_output_format == "h264") {
        codec = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
        extension = ".mp4";  // H.264 is a codec that needs a container (.mp4)
        std::cout << "Using H.264 codec in MP4 container" << std::endl;
    }
    else if (g_output_format == "yuv") {
        codec = 0; // Uncompressed YUV
        extension = ".avi";  // Use AVI container for YUV
        std::cout << "Using raw YUV format in AVI container" << std::endl;
    }
    else if (g_output_format == "avi") {
        codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        extension = ".avi";
        std::cout << "Using AVI format with MJPG codec" << std::endl;
    }
    else if (g_output_format == "mkv") {
        codec = cv::VideoWriter::fourcc('X', '2', '6', '4');
        extension = ".mkv";
        std::cout << "Using MKV format with X264 codec" << std::endl;
    }
    else {
        std::cout << "Unknown format '" << g_output_format << "', using default MP4" << std::endl;
        g_output_format = "mp4";
        codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        extension = ".mp4";
    }
    
    std::cout << "Video will be recorded at source frame rate: " << output_fps << " FPS" << std::endl;
    
    // For frame rate control
    double target_fps = g_using_super_res ? std::min(20.0, fps) : std::min(30.0, fps);
    double target_frame_time = 1000.0 / target_fps;
    
    // Simpler frame timing that works with standard C++ types
    auto last_frame_time = std::chrono::high_resolution_clock::now();
    
    while (g_running) {
        // Get processed frame - use non-blocking to check if frames are available
        timer.start("display_pop");
        bool success = buffer.popFrame(frame, false);
        timer.stop("display_pop");
        
        if (!success) {
            // No frame available, wait a bit and try again
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        
        // Initialize video writer when the first valid frame is available and saving is enabled
        if (g_save_video && !g_writer_initialized && !frame.empty()) {
            try {
                // Create output directory if it doesn't exist
                std::filesystem::path output_path = g_output_filename;
                
                // Check if the output filename already has an extension
                std::string current_ext = output_path.extension().string();
                if (current_ext.empty()) {
                    // Append the format extension if none exists
                    g_output_filename += extension;
                    output_path = g_output_filename;
                }
                
                std::filesystem::create_directories(output_path.parent_path());
                
                // Make sure we use an absolute path
                if (!output_path.is_absolute()) {
                    output_path = std::filesystem::absolute(output_path);
                    g_output_filename = output_path.string();
                }
                
                std::cout << "Creating video file: " << g_output_filename << std::endl;
                
                // Initialize video writer with the frame's properties and SOURCE FPS
                g_video_writer = std::make_unique<cv::VideoWriter>(
                    g_output_filename, 
                    codec, 
                    output_fps, 
                    cv::Size(frame.cols, frame.rows)
                );
                
                // Check if writer was properly initialized
                if (g_video_writer->isOpened()) {
                    g_writer_initialized = true;
                    std::cout << "Video recording started: " << g_output_filename << std::endl;
                    std::cout << "Output resolution: " << frame.cols << "x" << frame.rows << std::endl;
                    std::cout << "Output FPS: " << output_fps << " (matching source)" << std::endl;
                    std::cout << "Format: " << g_output_format << std::endl;
                } else {
                    std::cerr << "Failed to create video writer" << std::endl;
                    std::cerr << "Codec may not be supported on your system" << std::endl;
                    std::cerr << "Make sure the output directory exists and you have write permissions" << std::endl;
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
        
        // Display frame
        timer.start("display_show");
        cv::imshow("Video Feed", frame);
        timer.stop("display_show");
        
        g_frames_displayed++;
        
        // Enforce consistent frame rate for display (not affecting recording)
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_since_last = std::chrono::duration<double, std::milli>(current_time - last_frame_time).count();
        
        if (time_since_last < target_frame_time) {
            // Wait to maintain consistent frame rate
            std::this_thread::sleep_for(std::chrono::milliseconds(
                static_cast<int>(target_frame_time - time_since_last)));
        }
        
        // Update last frame time
        last_frame_time = std::chrono::high_resolution_clock::now();
        
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

// Super-resolution background thread function
void sr_thread_function(std::shared_ptr<DnnSuperRes> dnn_sr, int target_width, int target_height) {
    std::cout << "Super-resolution background thread started" << std::endl;
    
    while (g_running) {
        cv::Mat frame_to_process;
        bool got_frame = false;
        
        // Get a frame from the queue
        {
            std::lock_guard<std::mutex> lock(g_sr_queue_mutex);
            if (!g_sr_frame_queue.empty()) {
                frame_to_process = g_sr_frame_queue.front();
                g_sr_frame_queue.pop();
                got_frame = true;
                g_sr_processing_active = true;
            }
        }
        
        if (got_frame) {
            // Process with SR
            cv::Mat sr_output;
            auto start_time = std::chrono::high_resolution_clock::now();
            
            bool success = dnn_sr->upscale(frame_to_process, sr_output);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            double processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            
            if (success) {
                // Store the result for main thread to use
                std::lock_guard<std::mutex> lock(g_sr_result_mutex);
                g_last_sr_result = sr_output.clone();
                std::cout << "Background SR completed in " << processing_time << " ms" << std::endl;
            }
            
            g_sr_processing_active = false;
        } else {
            // No frames to process, sleep to reduce CPU usage
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
    
    std::cout << "Super-resolution background thread finished" << std::endl;
}

// Add this function
void addMotionBlur(cv::Mat& frame, float strength = 0.5) {
    cv::Mat blurred;
    cv::GaussianBlur(frame, blurred, cv::Size(0, 0), 3);
    cv::addWeighted(frame, 1.0 - strength, blurred, strength, 0, frame);
}

int main(int argc, char* argv[]) {
    std::cout << "Low-Latency Video Processing System" << std::endl;
    
    // Register signal handler for clean shutdown
    signal(SIGINT, signalHandler);  // Ctrl+C
    signal(SIGTERM, signalHandler); // termination request
    
    // Check if a video file path is provided as a command-line argument
    std::string video_source;
    bool use_video_file = false;
    bool simulate_realtime = true; // New flag to control frame rate simulation
    bool use_super_res = false;    // Default to bicubic upscaling
    int target_width = 1920;       // Default output width (reduced from 1920)
    int target_height = 1080;       // Default output height (reduced from 1080)
    std::string g_output_format = "mp4"; // Default format
    
    // Declare algorithm variable before command-line parsing
    Upscaler::Algorithm algorithm = Upscaler::BICUBIC; // Default algorithm
    
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
        } else if (arg == "--fast" || arg == "-f") {
            simulate_realtime = false;
            std::cout << "Fast processing mode enabled (no frame rate control)" << std::endl;
        } else if (arg == "--super-res" || arg == "-sr") {
            use_super_res = true;
            algorithm = Upscaler::SUPER_RES; // Set algorithm here
            std::cout << "Super-resolution upscaling enabled" << std::endl;
        } else if (arg == "--realesrgan") {
            use_super_res = true;
            algorithm = Upscaler::REAL_ESRGAN; // Set algorithm to REAL_ESRGAN
            std::cout << "RealESRGAN super-resolution upscaling enabled" << std::endl;
        } else if (arg == "--resolution" || arg == "-res") {
            if (i + 2 < argc) {
                target_width = std::stoi(argv[++i]);
                target_height = std::stoi(argv[++i]);
                std::cout << "Output resolution set to " << target_width << "x" << target_height << std::endl;
            }
        // }else if (arg == "--no-temporal") {
        //     upscaler.setUseTemporalConsistency(false);
        //     std::cout << "Temporal consistency disabled" << std::endl;
        // }
        // else if (arg == "--no-sharpening") {
        //     upscaler.setUseAdaptiveSharpening(false);
        //     std::cout << "Adaptive sharpening disabled" << std::endl;
        // }
        // else if (arg == "--no-bilateral") {
        //     upscaler.setUseSelectiveBilateral(false);
        //     std::cout << "Selective bilateral filtering disabled" << std::endl;
        // }
        // else if (arg == "--animation-mode") {
        //     // Optimize parameters for animation content
        //     if (upscaler.getBilateralPreProcessor()) {
        //         auto config = upscaler.getBilateralPreProcessor()->getConfig();
        //         config.detail_threshold = 20.0; // Lower threshold to preserve more details
        //         config.edge_preserve = 2.5;     // Stronger edge preservation
        //         upscaler.getBilateralPreProcessor()->setConfig(config);
        //     }
            
        //     if (upscaler.getAdaptiveSharpening()) {
        //         auto config = upscaler.getAdaptiveSharpening()->getConfig();
        //         config.strength = 0.6f;         // Reduced overall sharpening
        //         config.edge_strength = 1.0f;    // Standard edge sharpening
        //         upscaler.getAdaptiveSharpening()->setConfig(config);
        //     }
            
        //     if (upscaler.getTemporalConsistency()) {
        //         auto config = upscaler.getTemporalConsistency()->getConfig();
        //         config.blend_strength = 0.8f;   // Stronger temporal blending
        //         config.scene_change_threshold = 80.0f; // Lower threshold for scene changes
        //         upscaler.getTemporalConsistency()->setConfig(config);
        //     }
            
        //     std::cout << "Animation mode enabled with optimized parameters" << std::endl;
        // }
        // else if (arg == "--live-action-mode") {
        //     // Optimize parameters for live-action content
        //     if (upscaler.getBilateralPreProcessor()) {
        //         auto config = upscaler.getBilateralPreProcessor()->getConfig();
        //         config.detail_threshold = 35.0; // Higher threshold to reduce noise
        //         config.sigma_color = 35.0;      // Stronger color smoothing
        //         upscaler.getBilateralPreProcessor()->setConfig(config);
        //     }
            
        //     if (upscaler.getAdaptiveSharpening()) {
        //         auto config = upscaler.getAdaptiveSharpening()->getConfig();
        //         config.strength = 0.9f;         // Stronger overall sharpening
        //         config.edge_strength = 1.3f;    // Stronger edge sharpening
        //         upscaler.getAdaptiveSharpening()->setConfig(config);
        //     }
            
        //     if (upscaler.getTemporalConsistency()) {
        //         auto config = upscaler.getTemporalConsistency()->getConfig();
        //         config.blend_strength = 0.5f;   // Moderate temporal blending
        //         config.motion_threshold = 20.0f; // Higher motion threshold
        //         upscaler.getTemporalConsistency()->setConfig(config);
        //     }
        } else if (arg == "--format" || arg == "-fmt") {
            if (i + 1 < argc) {
                g_output_format = argv[++i];
                std::transform(g_output_format.begin(), g_output_format.end(), 
                             g_output_format.begin(), ::tolower);
                std::cout << "Output format set to: " << g_output_format << std::endl;
            }
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
            std::cerr << "Usage: " << argv[0] << " [camera_index|video_file_path] [--output filename] [--record] [--fast] [--super-res] [--realesrgan] [--resolution width height] [--format format]" << std::endl;
            std::cerr << "Supported formats: mp4, h264, yuv, avi, mkv" << std::endl;
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
    
    // Initialize camera or video source with lower resolution for better performance
    int capture_width = 640;
    int capture_height = 360; // Lower than the previous 480 for better performance
    int capture_fps = 30;
    
    if (!source->initialize(capture_width, capture_height, capture_fps)) {
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
    
    // Use the algorithm variable we already set (don't redeclare it)
    std::cout << "Using " << (use_super_res ? "Super-Resolution" : "Bicubic") 
              << " upscaling algorithm" << std::endl;
    
    // Create upscaler with target resolution and chosen algorithm
    Upscaler upscaler(algorithm, true);
    if (!upscaler.initialize(target_width, target_height)) {
        std::cerr << "Error: Could not initialize upscaler" << std::endl;
        return -1;
    }
    
    std::cout << "Upscaler initialized with algorithm: " 
              << upscaler.getAlgorithmName()
              << ", using " << (upscaler.isUsingGPU() ? "GPU" : "CPU") << std::endl;
    
    // Create frame buffers with sizes based on algorithm
    // Use much larger buffers for super-res to prevent drops
    int raw_buffer_size = use_super_res ? 120 : 60;
    int processed_buffer_size = use_super_res ? 90 : 60;
    
    FrameBuffer raw_buffer(raw_buffer_size);
    FrameBuffer processed_buffer(processed_buffer_size);
    
    std::cout << "Frame buffers initialized with sizes " << raw_buffer_size 
              << " and " << processed_buffer_size << std::endl;
    
    // Create timer for performance measurement
    Timer timer;
    
    // For video files, adjust playback rate based on selected algorithm
    double playback_rate = 1.0;
    if (use_video_file && use_super_res) {
        // Slow down playback for super-res to avoid frame drops
        playback_rate = 0.25;
        std::cout << "Video playback rate set to " << playback_rate 
                  << "x due to super-resolution processing" << std::endl;
    }
    
    // Calculate target FPS for frame rate control
    double target_fps = (use_video_file && simulate_realtime) ? 
                         source_fps * playback_rate : 0.0;
    
    // Start all threads
    std::cout << "Starting pipeline threads..." << std::endl;
    
    std::thread capture(capture_thread, std::ref(*source), std::ref(raw_buffer), 
                         std::ref(timer), use_video_file, target_fps, use_super_res);
    std::thread processor(processing_thread, std::ref(raw_buffer), std::ref(processed_buffer), 
                         std::ref(upscaler), std::ref(timer));
    std::thread display(displayLoop, std::ref(processed_buffer), std::ref(timer), 
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