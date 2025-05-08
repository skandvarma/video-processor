#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>

#include "temporal_consistency.h"
#include "adaptive_sharpening.h"
#include "selective_bilateral.h"
#include "dnn_super_res.h"
#include "timer.h"

// Helper function for timing
double measureTime(const std::function<void()>& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Function to display images side by side
void displaySideBySide(const cv::Mat& img1, const cv::Mat& img2, const std::string& window_name) {
    int height = std::max(img1.rows, img2.rows);
    int width = img1.cols + img2.cols;
    cv::Mat display(height, width, img1.type(), cv::Scalar(0, 0, 0));
    
    img1.copyTo(display(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(display(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));
    
    cv::imshow(window_name, display);
}

// Function to save enhancement result
void saveEnhancementResult(const std::string& prefix, const cv::Mat& input, const cv::Mat& output) {
    static int counter = 0;
    std::string input_filename = prefix + "_input_" + std::to_string(counter) + ".png";
    std::string output_filename = prefix + "_output_" + std::to_string(counter) + ".png";
    
    cv::imwrite(input_filename, input);
    cv::imwrite(output_filename, output);
    
    counter++;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    std::string input_path;
    std::string output_path = "enhanced_output.mp4";
    bool use_video = false;
    bool use_camera = false;
    int camera_index = 0;
    bool save_results = false;
    std::string mode = "default";
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--camera" || arg == "-c") {
            use_camera = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                camera_index = std::stoi(argv[++i]);
            }
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                output_path = argv[++i];
            }
        } else if (arg == "--save" || arg == "-s") {
            save_results = true;
        } else if (arg == "--mode" || arg == "-m") {
            if (i + 1 < argc) {
                mode = argv[++i];
            }
        } else {
            // Assume it's the input path
            input_path = arg;
            use_video = true;
        }
    }
    
    if (!use_camera && !use_video) {
        std::cout << "Usage: " << argv[0] << " [video_file] [options]" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  --camera, -c [index]   Use camera instead of video file" << std::endl;
        std::cout << "  --output, -o [path]    Output path for processed video" << std::endl;
        std::cout << "  --save, -s             Save frame-by-frame results" << std::endl;
        std::cout << "  --mode, -m [mode]      Processing mode: default, animation, live-action, film, low-quality" << std::endl;
        return 0;
    }
    
    // Initialize video capture
    cv::VideoCapture cap;
    if (use_camera) {
        cap.open(camera_index);
        std::cout << "Opening camera " << camera_index << std::endl;
    } else if (use_video) {
        cap.open(input_path);
        std::cout << "Opening video file: " << input_path << std::endl;
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source" << std::endl;
        return -1;
    }
    
    // Get video properties
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0; // Default if not available
    
    std::cout << "Video properties: " << frame_width << "x" << frame_height 
              << " @ " << fps << " FPS" << std::endl;
    
    // Calculate target dimensions (2x upscale)
    int target_width = frame_width * 2;
    int target_height = frame_height * 2;
    
    // Initialize output video writer
    cv::VideoWriter writer;
    if (save_results) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(output_path, fourcc, fps, cv::Size(target_width, target_height));
        
        if (!writer.isOpened()) {
            std::cerr << "Error: Could not create output video" << std::endl;
            return -1;
        }
        
        std::cout << "Writing output to: " << output_path << std::endl;
    }
    
    // Initialize enhancement modules
    bool use_gpu = true;
    
    // 1. Selective Bilateral Filtering (Pre-processing)
    SelectiveBilateral bilateral_pre(
        SelectiveBilateral::Config{
            SelectiveBilateral::PRE_PROCESSING,
            use_gpu,
            true  // adaptive params
        });
    bilateral_pre.initialize();
    
    // 2. Super-Resolution
    DnnSuperRes superres(
        "models/RRDB_ESRGAN_x4.onnx", 
        "esrgan", 
        4, 
        DnnSuperRes::REAL_ESRGAN);
    superres.setTargetSize(target_width, target_height);
    superres.setUseGPU(use_gpu);
    if (!superres.initialize()) {
        std::cerr << "Warning: Failed to initialize super-resolution. "
                  << "Check if model file exists in the models directory." << std::endl;
        std::cerr << "Falling back to bicubic upscaling." << std::endl;
    }
    
    // 3. Adaptive Sharpening
    AdaptiveSharpening sharpening(
        AdaptiveSharpening::Config{
            0.8f,  // strength
            1.2f,  // edge strength
            0.4f,  // smooth strength
            30.0f, // edge threshold
            1.5f,  // sigma
            5,     // kernel size
            true,  // preserve tone
            use_gpu
        });
    sharpening.initialize();
    
    // 4. Selective Bilateral Filtering (Post-processing)
    SelectiveBilateral bilateral_post(
        SelectiveBilateral::Config{
            SelectiveBilateral::POST_PROCESSING,
            use_gpu,
            true  // adaptive params
        });
    bilateral_post.initialize();
    
    // 5. Temporal Consistency
    TemporalConsistency temporal_consistency(
        TemporalConsistency::Config{
            3,     // buffer size
            0.6f,  // blend strength
            15.0f, // motion threshold
            100.0f, // scene change threshold
            use_gpu
        });
    temporal_consistency.initialize();
    
    // Apply mode-specific settings
    if (mode == "animation") {
        std::cout << "Using animation-optimized parameters" << std::endl;
        
        // Get configuration objects
        auto tc_config = temporal_consistency.getConfig();
        auto as_config = sharpening.getConfig();
        auto bp_config = bilateral_pre.getConfig();
        auto bpo_config = bilateral_post.getConfig();
        
        // Modify for animation
        tc_config.blend_strength = 0.75f;
        tc_config.motion_threshold = 12.0f;
        
        as_config.strength = 0.6f;
        as_config.edge_threshold = 20.0f;
        
        bp_config.diameter = 5;
        bp_config.sigma_color = 25.0;
        bp_config.sigma_space = 25.0;
        bp_config.detail_threshold = 20.0;
        bp_config.edge_preserve = 2.5;
        
        bpo_config.diameter = 3;
        bpo_config.detail_threshold = 15.0;
        bpo_config.edge_preserve = 3.0;
        
        // Apply modified configurations
        temporal_consistency.setConfig(tc_config);
        sharpening.setConfig(as_config);
        bilateral_pre.setConfig(bp_config);
        bilateral_post.setConfig(bpo_config);
    }
    else if (mode == "live-action") {
        std::cout << "Using live-action optimized parameters" << std::endl;
        
        // Get configuration objects
        auto tc_config = temporal_consistency.getConfig();
        auto as_config = sharpening.getConfig();
        auto bp_config = bilateral_pre.getConfig();
        auto bpo_config = bilateral_post.getConfig();
        
        // Modify for live-action
        tc_config.buffer_size = 4;
        tc_config.blend_strength = 0.5f;
        tc_config.motion_threshold = 20.0f;
        
        as_config.strength = 0.9f;
        as_config.edge_strength = 1.3f;
        
        bp_config.diameter = 7;
        bp_config.sigma_color = 35.0;
        bp_config.sigma_space = 35.0;
        
        // Apply modified configurations
        temporal_consistency.setConfig(tc_config);
        sharpening.setConfig(as_config);
        bilateral_pre.setConfig(bp_config);
        bilateral_post.setConfig(bpo_config);
    }
    else if (mode == "film") {
        std::cout << "Using film restoration optimized parameters" << std::endl;
        
        // Get configuration objects
        auto tc_config = temporal_consistency.getConfig();
        auto as_config = sharpening.getConfig();
        auto bp_config = bilateral_pre.getConfig();
        
        // Modify for film restoration
        tc_config.buffer_size = 5;
        tc_config.blend_strength = 0.8f;
        tc_config.motion_threshold = 10.0f;
        
        as_config.strength = 1.0f;
        as_config.edge_strength = 1.5f;
        
        bp_config.diameter = 9;
        bp_config.sigma_color = 45.0;
        bp_config.sigma_space = 45.0;
        
        // Apply modified configurations
        temporal_consistency.setConfig(tc_config);
        sharpening.setConfig(as_config);
        bilateral_pre.setConfig(bp_config);
    }
    else if (mode == "low-quality") {
        std::cout << "Using low-quality source optimized parameters" << std::endl;
        
        // Get configuration objects
        auto tc_config = temporal_consistency.getConfig();
        auto as_config = sharpening.getConfig();
        auto bp_config = bilateral_pre.getConfig();
        
        // Modify for low-quality sources
        tc_config.blend_strength = 0.7f;
        tc_config.motion_threshold = 25.0f;
        
        as_config.strength = 0.5f;
        as_config.edge_strength = 0.8f;
        
        bp_config.diameter = 11;
        bp_config.sigma_color = 50.0;
        bp_config.sigma_space = 50.0;
        bp_config.detail_threshold = 45.0;
        
        // Apply modified configurations
        temporal_consistency.setConfig(tc_config);
        sharpening.setConfig(as_config);
        bilateral_pre.setConfig(bp_config);
    }
    
    // Create timer
    Timer timer;
    
    // Main processing loop
    cv::Mat frame, preprocessed, upscaled, sharpened, postprocessed, enhanced;
    int frame_count = 0;
    double total_time = 0.0;
    
    std::cout << "Press 'q' to quit, 's' to save the current frame" << std::endl;
    std::cout << "Press '1' to toggle bilateral pre-processing" << std::endl;
    std::cout << "Press '2' to toggle sharpening" << std::endl;
    std::cout << "Press '3' to toggle bilateral post-processing" << std::endl;
    std::cout << "Press '4' to toggle temporal consistency" << std::endl;
    
    bool use_bilateral_pre = true;
    bool use_sharpening = true;
    bool use_bilateral_post = true;
    bool use_temporal = true;
    
    while (true) {
        // Read a new frame
        if (!cap.read(frame)) {
            std::cout << "End of video or error reading frame" << std::endl;
            break;
        }
        
        frame_count++;
        
        // Start timing
        timer.start("total");
        
        // 1. Pre-processing with selective bilateral filtering
        timer.start("bilateral_pre");
        if (use_bilateral_pre) {
            bilateral_pre.process(frame, preprocessed);
        } else {
            frame.copyTo(preprocessed);
        }
        timer.stop("bilateral_pre");
        
        // 2. Super-resolution upscaling
        timer.start("superres");
        if (superres.isInitialized()) {
            superres.upscale(preprocessed, upscaled);
        } else {
            // Fallback to bicubic
            cv::resize(preprocessed, upscaled, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
        }
        timer.stop("superres");
        
        // 3. Adaptive sharpening
        timer.start("sharpening");
        if (use_sharpening) {
            sharpening.process(upscaled, sharpened);
        } else {
            upscaled.copyTo(sharpened);
        }
        timer.stop("sharpening");
        
        // 4. Post-processing with selective bilateral filtering
        timer.start("bilateral_post");
        if (use_bilateral_post) {
            bilateral_post.process(sharpened, postprocessed);
        } else {
            sharpened.copyTo(postprocessed);
        }
        timer.stop("bilateral_post");
        
        // 5. Temporal consistency
        timer.start("temporal");
        if (use_temporal) {
            temporal_consistency.process(postprocessed, enhanced);
        } else {
            postprocessed.copyTo(enhanced);
        }
        timer.stop("temporal");
        
        // Stop total timing
        timer.stop("total");
        total_time += timer.getDuration("total");
        
        // Display results
        cv::resize(frame, frame, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC); // Scale to match output
        displaySideBySide(frame, enhanced, "Original vs. Enhanced");
        
        // Print performance info every 30 frames
        if (frame_count % 30 == 0) {
            std::cout << "\n=== Frame " << frame_count << " ===\n";
            std::cout << "Average processing time: " << total_time / frame_count << " ms" << std::endl;
            timer.printStats();
        }
        
        // Write to output video if enabled
        if (writer.isOpened()) {
            writer.write(enhanced);
        }
        
        // Handle keyboard input
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) { // 'q' or ESC
            break;
        } else if (key == 's') {
            // Save current frame
            std::string filename = "enhanced_frame_" + std::to_string(frame_count) + ".png";
            cv::imwrite(filename, enhanced);
            std::cout << "Saved frame to " << filename << std::endl;
        } else if (key == '1') {
            use_bilateral_pre = !use_bilateral_pre;
            std::cout << "Bilateral pre-processing: " << (use_bilateral_pre ? "ON" : "OFF") << std::endl;
        } else if (key == '2') {
            use_sharpening = !use_sharpening;
            std::cout << "Adaptive sharpening: " << (use_sharpening ? "ON" : "OFF") << std::endl;
        } else if (key == '3') {
            use_bilateral_post = !use_bilateral_post;
            std::cout << "Bilateral post-processing: " << (use_bilateral_post ? "ON" : "OFF") << std::endl;
        } else if (key == '4') {
            use_temporal = !use_temporal;
            std::cout << "Temporal consistency: " << (use_temporal ? "ON" : "OFF") << std::endl;
        }
    }
    
    // Release resources
    cap.release();
    if (writer.isOpened()) {
        writer.release();
    }
    cv::destroyAllWindows();
    
    // Print final statistics
    std::cout << "\n=== Final Statistics ===\n";
    std::cout << "Total frames processed: " << frame_count << std::endl;
    std::cout << "Average processing time: " << total_time / frame_count << " ms" << std::endl;
    std::cout << "Effective FPS: " << 1000.0 / (total_time / frame_count) << std::endl;
    timer.printStats();
    
    return 0;
}