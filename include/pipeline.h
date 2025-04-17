#pragma once

#include "upscaler.h"
#include "timer.h"

#include <string>
#include <memory>
#include <atomic>

/**
 * @brief Integrated pipeline that connects all video processing components
 * 
 * This class orchestrates the overall video processing workflow by managing
 * multiple threads, coordinating data flow between components, and optimizing
 * for low latency.
 */
class Pipeline {
public:
    /**
     * @brief Configuration options for the pipeline
     */
    struct Config {
        // Camera options
        int camera_index = 0;
        std::string video_source = ""; // Empty means use camera_index instead
        int camera_width = 1280;
        int camera_height = 720;
        int camera_fps = 60;
        
        // Upscaler options
        int target_width = 1920;
        int target_height = 1080;
        Upscaler::Algorithm upscale_algorithm = Upscaler::BILINEAR;
        bool use_gpu = true;
        
        // Buffer options
        int buffer_size = 5;
        
        // Display options
        std::string window_name = "Video Output";
        bool show_metrics = true;
        bool enable_vsync = false;
        int max_display_fps = 60;
        
        // Performance options
        bool measure_latency = true;
    };
    
    /**
     * @brief Construct a new Pipeline with default configuration
     */
    Pipeline();
    
    /**
     * @brief Construct a new Pipeline with custom configuration
     * @param config Configuration options
     */
    Pipeline(const Config& config);
    
    /**
     * @brief Destroy the Pipeline and free resources
     */
    ~Pipeline();
    
    /**
     * @brief Initialize all pipeline components
     * @param camera_index Index of the camera to use
     * @return true if initialization was successful
     */
    bool initialize(int camera_index = 0);
    
    /**
     * @brief Initialize all pipeline components with a video file
     * @param video_path Path to the video file
     * @return true if initialization was successful
     */
    bool initialize(const std::string& video_path);
    
    /**
     * @brief Start the pipeline processing
     * @return true if started successfully
     */
    bool start();
    
    /**
     * @brief Stop the pipeline processing
     */
    void stop();
    
    /**
     * @brief Check if the pipeline is currently running
     * @return true if pipeline is running
     */
    bool isRunning() const;
    
    /**
     * @brief Wait for a key press to interrupt the pipeline
     * @param key The key code to wait for (default: 'q')
     * @return true if the specified key was pressed
     */
    bool waitForKey(int key = 'q');
    
    /**
     * @brief Get end-to-end latency of the pipeline in milliseconds
     * @return Latency in milliseconds
     */
    double getLatency() const;
    
    /**
     * @brief Get the effective frames per second of the pipeline
     * @return Frames per second
     */
    double getFPS() const;
    
    /**
     * @brief Set the target resolution for upscaling
     * @param width Target width
     * @param height Target height
     */
    void setTargetResolution(int width, int height);
    
    /**
     * @brief Set the buffer size for the pipeline
     * @param size Number of frames to buffer
     */
    void setBufferSize(int size);
    
    /**
     * @brief Set display options
     * @param show_metrics Whether to show performance metrics
     */
    void setDisplayOptions(bool show_metrics);
    
    /**
     * @brief Print performance statistics to console
     */
    void printPerformanceStats() const;
    
private:
    // Using PIMPL (Pointer to Implementation) pattern to hide implementation details
    // and avoid compilation dependency issues
    class Impl;
    std::unique_ptr<Impl> m_impl;
    
    // Performance metrics that need to be accessible from outside
    std::atomic<double> m_latency{0.0};
    std::atomic<double> m_fps{0.0};
    Timer m_timer;
    Config m_config;
};