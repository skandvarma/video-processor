#include "pipeline.h"
#include "camera.h"  // We still need this include for the Camera class
#include <iostream>
#include <string>
#include <thread>
#include <csignal>
#include <algorithm>

// Global flag for signal handling
volatile std::sig_atomic_t g_shutdown_requested = 0;

// Signal handler
void signal_handler(int signal) {
    g_shutdown_requested = 1;
}

// Function to display help
void display_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -h, --help               Display this help message" << std::endl;
    std::cout << "  -c, --camera INDEX       Specify camera index (default: 0)" << std::endl;
    std::cout << "  -r, --resolution WxH     Set camera resolution (default: 1280x720)" << std::endl;
    std::cout << "  -t, --target WxH         Set target resolution (default: 1920x1080)" << std::endl;
    std::cout << "  -a, --algorithm ALGO     Set upscale algorithm (nearest, bilinear, bicubic, lanczos)" << std::endl;
    std::cout << "  -g, --gpu [on|off]       Enable/disable GPU acceleration" << std::endl;
    std::cout << "  -b, --buffer SIZE        Set buffer size (default: 5)" << std::endl;
    std::cout << "  -v, --vsync [on|off]     Enable/disable VSync" << std::endl;
    std::cout << "  -f, --fps FPS            Set maximum display FPS (default: 60)" << std::endl;
    std::cout << "  -m, --metrics [on|off]   Show/hide performance metrics" << std::endl;
}

// Function to parse resolution string (e.g., "1920x1080")
bool parse_resolution(const std::string& res_str, int& width, int& height) {
    size_t pos = res_str.find('x');
    if (pos == std::string::npos) {
        return false;
    }
    
    try {
        width = std::stoi(res_str.substr(0, pos));
        height = std::stoi(res_str.substr(pos + 1));
        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

// Function to parse algorithm string
Upscaler::Algorithm parse_algorithm(const std::string& algo_str) {
    if (algo_str == "nearest") return Upscaler::NEAREST;
    if (algo_str == "bilinear") return Upscaler::BILINEAR;
    if (algo_str == "bicubic") return Upscaler::BICUBIC;
    if (algo_str == "lanczos") return Upscaler::LANCZOS;
    if (algo_str == "superres") return Upscaler::SUPER_RES;
    
    // Default to bilinear
    return Upscaler::BILINEAR;
}

// Parse boolean option (on/off, true/false, 1/0)
bool parse_bool_option(const std::string& value, bool default_value) {
    if (value == "on" || value == "true" || value == "1") return true;
    if (value == "off" || value == "false" || value == "0") return false;
    return default_value;
}

int main(int argc, char* argv[]) {
    std::cout << "Low-Latency Video Processing System - Phase 4" << std::endl;
    
    // Install signal handler for clean shutdown
    std::signal(SIGINT, signal_handler);
    
    // Check for help flag
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            display_help(argv[0]);
            return 0;
        }
    }
    
    // Create default pipeline configuration
    Pipeline::Config config;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-c" || arg == "--camera") {
            if (i + 1 < argc) {
                config.camera_index = std::stoi(argv[++i]);
            }
        }
        else if (arg == "-r" || arg == "--resolution") {
            if (i + 1 < argc) {
                if (parse_resolution(argv[++i], config.camera_width, config.camera_height)) {
                    std::cout << "Camera resolution set to: " 
                              << config.camera_width << "x" << config.camera_height << std::endl;
                }
            }
        }
        else if (arg == "-t" || arg == "--target") {
            if (i + 1 < argc) {
                if (parse_resolution(argv[++i], config.target_width, config.target_height)) {
                    std::cout << "Target resolution set to: " 
                              << config.target_width << "x" << config.target_height << std::endl;
                }
            }
        }
        else if (arg == "-a" || arg == "--algorithm") {
            if (i + 1 < argc) {
                config.upscale_algorithm = parse_algorithm(argv[++i]);
                std::cout << "Upscaling algorithm set" << std::endl;
            }
        }
        else if (arg == "-g" || arg == "--gpu") {
            if (i + 1 < argc) {
                config.use_gpu = parse_bool_option(argv[++i], true);
                std::cout << "GPU acceleration: " << (config.use_gpu ? "enabled" : "disabled") << std::endl;
            }
        }
        else if (arg == "-b" || arg == "--buffer") {
            if (i + 1 < argc) {
                config.buffer_size = std::stoi(argv[++i]);
                std::cout << "Buffer size set to: " << config.buffer_size << std::endl;
            }
        }
        else if (arg == "-v" || arg == "--vsync") {
            if (i + 1 < argc) {
                config.enable_vsync = parse_bool_option(argv[++i], false);
                std::cout << "VSync: " << (config.enable_vsync ? "enabled" : "disabled") << std::endl;
            }
        }
        else if (arg == "-f" || arg == "--fps") {
            if (i + 1 < argc) {
                config.max_display_fps = std::stoi(argv[++i]);
                std::cout << "Max display FPS set to: " << config.max_display_fps << std::endl;
            }
        }
        else if (arg == "-m" || arg == "--metrics") {
            if (i + 1 < argc) {
                config.show_metrics = parse_bool_option(argv[++i], true);
                std::cout << "Performance metrics: " << (config.show_metrics ? "shown" : "hidden") << std::endl;
            }
        }
    }
    
    // List available cameras
    std::cout << "Checking available cameras..." << std::endl;
    auto available_cameras = Camera::listAvailableCameras();
    
    if (available_cameras.empty()) {
        std::cerr << "No cameras detected! Please connect a camera and try again." << std::endl;
        return -1;
    }
    
    // Use the first available camera if specified camera is not available
    if (std::find(available_cameras.begin(), available_cameras.end(), config.camera_index) 
        == available_cameras.end()) {
        std::cout << "Camera index " << config.camera_index << " not available." << std::endl;
        config.camera_index = available_cameras[0];
        std::cout << "Using camera index " << config.camera_index << " instead." << std::endl;
    }
    
    // Create pipeline with configuration
    Pipeline pipeline(config);
    
    // Initialize pipeline
    if (!pipeline.initialize()) {
        std::cerr << "Failed to initialize pipeline" << std::endl;
        return -1;
    }
    
    // Start pipeline
    if (!pipeline.start()) {
        std::cerr << "Failed to start pipeline" << std::endl;
        return -1;
    }
    
    std::cout << "Pipeline running. Press 'q' to quit." << std::endl;
    
    // Wait for quit signal
    bool quit_requested = false;
    while (!quit_requested && !g_shutdown_requested) {
        // Check for 'q' key press
        if (pipeline.waitForKey('q')) {
            quit_requested = true;
        }
        
        // Print performance stats every 5 seconds
        static auto last_stats_time = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - last_stats_time).count() >= 5) {
            pipeline.printPerformanceStats();
            last_stats_time = now;
        }
        
        // Small delay to avoid hammering the CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Stop the pipeline
    pipeline.stop();
    
    std::cout << "Pipeline shutdown complete. Final statistics:" << std::endl;
    pipeline.printPerformanceStats();
    
    return 0;
}