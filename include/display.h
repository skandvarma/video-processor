#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>
#include <mutex>
#include "timer.h"

/**
 * @brief Display component for rendering processed frames with minimal latency
 * 
 * This class handles the efficient rendering of processed video frames
 * with options for performance monitoring and latency optimization.
 */
class Display {
public:
    /**
     * @brief Construct a new Display object
     * @param width Display window width
     * @param height Display window height
     */
    Display(int width = 1920, int height = 1080);
    
    /**
     * @brief Destroy the Display object and free resources
     */
    ~Display();
    
    /**
     * @brief Initialize the display component
     * @param window_name Name of the display window
     * @return true if initialization was successful
     */
    bool initialize(const std::string& window_name = "Video Output");
    
    /**
     * @brief Render a frame to the display
     * @param frame The frame to render
     * @return true if rendering was successful
     */
    bool renderFrame(const cv::Mat& frame);
    
    /**
     * @brief Enable or disable performance metrics overlay
     * @param show Whether to show metrics
     */
    void showPerformanceMetrics(bool show);
    
    /**
     * @brief Clean up display resources
     */
    void cleanup();
    
    /**
     * @brief Enable or disable vertical sync
     * @param enabled Whether to enable VSync
     */
    void setVSync(bool enabled);
    
    /**
     * @brief Set maximum frame rate for display
     * @param fps Maximum frames per second (0 for unlimited)
     */
    void setMaxFrameRate(int fps);
    
    /**
     * @brief Get the last measured render time in milliseconds
     * @return Render time in milliseconds
     */
    double getLastRenderTime() const;
    
    /**
     * @brief Get the current display FPS
     * @return Current FPS
     */
    double getCurrentFPS() const;
    
private:
    // Rendering state
    std::string m_window_name;
    int m_width;
    int m_height;
    bool m_show_metrics;
    bool m_vsync_enabled;
    int m_max_fps;
    
    // Performance tracking
    Timer m_display_timer;
    double m_last_render_time;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_last_frame_time;
    std::atomic<double> m_current_fps;
    
    // Frame rate control
    std::chrono::microseconds m_frame_interval;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_next_frame_time;
    
    // Thread synchronization
    std::mutex m_render_mutex;
    
    // Internal methods
    void drawPerformanceOverlay(cv::Mat& frame);
    void updateFPS();
    void limitFrameRate();
};