#include "display.h"
#include <iostream>
#include <iomanip>
#include <thread>

Display::Display(int width, int height)
    : m_width(width),
      m_height(height),
      m_show_metrics(true),
      m_vsync_enabled(false),
      m_max_fps(60),
      m_last_render_time(0.0),
      m_current_fps(0.0) {
    
    // Calculate frame interval in microseconds based on max FPS
    m_frame_interval = std::chrono::microseconds(static_cast<int>(1000000.0 / m_max_fps));
    m_next_frame_time = std::chrono::high_resolution_clock::now();
    m_last_frame_time = m_next_frame_time;
}

Display::~Display() {
    cleanup();
}

bool Display::initialize(const std::string& window_name) {
    m_window_name = window_name;
    
    // Create window with OpenCV
    try {
        cv::namedWindow(m_window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(m_window_name, m_width, m_height);
        
        // Set window properties to minimize latency
        // Note: OpenCV doesn't have direct VSync control, but we'll simulate it
        
        std::cout << "Display initialized: " << m_width << "x" << m_height 
                  << " @ " << m_max_fps << " FPS" << std::endl;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Failed to initialize display: " << e.what() << std::endl;
        return false;
    }
}

bool Display::renderFrame(const cv::Mat& frame) {
    std::lock_guard<std::mutex> lock(m_render_mutex);
    
    if (frame.empty()) {
        std::cerr << "Warning: Attempted to render empty frame" << std::endl;
        return false;
    }
    
    // Limit frame rate if necessary
    limitFrameRate();
    
    // Start timing the render operation
    m_display_timer.start("render");
    
    // Clone the frame to avoid modifying the original if showing metrics
    cv::Mat display_frame;
    if (m_show_metrics) {
        frame.copyTo(display_frame);
        drawPerformanceOverlay(display_frame);
    } else {
        display_frame = frame;
    }
    
    // Display the frame
    try {
        cv::imshow(m_window_name, display_frame);
        
        // Process window events (short wait to allow GUI to update)
        cv::waitKey(1);
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error rendering frame: " << e.what() << std::endl;
        m_display_timer.stop("render");
        return false;
    }
    
    // Stop timing and record
    m_display_timer.stop("render");
    m_last_render_time = m_display_timer.getDuration("render");
    
    // Update FPS calculation
    updateFPS();
    
    return true;
}

void Display::showPerformanceMetrics(bool show) {
    m_show_metrics = show;
}

void Display::cleanup() {
    try {
        cv::destroyWindow(m_window_name);
    }
    catch (...) {
        // Ignore errors during cleanup
    }
}

void Display::setVSync(bool enabled) {
    m_vsync_enabled = enabled;
    
    // OpenCV doesn't provide direct VSync control
    // In a more advanced implementation, this could interface with
    // OpenGL/DirectX rendering contexts for true VSync control
    std::cout << "VSync " << (enabled ? "enabled" : "disabled") << std::endl;
}

void Display::setMaxFrameRate(int fps) {
    m_max_fps = (fps > 0) ? fps : 0;
    
    if (m_max_fps > 0) {
        m_frame_interval = std::chrono::microseconds(static_cast<int>(1000000.0 / m_max_fps));
    }
    
    std::cout << "Display max framerate set to " 
              << (m_max_fps > 0 ? std::to_string(m_max_fps) : "unlimited") << std::endl;
}

double Display::getLastRenderTime() const {
    return m_last_render_time;
}

double Display::getCurrentFPS() const {
    return m_current_fps.load();
}

void Display::drawPerformanceOverlay(cv::Mat& frame) {
    // Add display statistics
    std::stringstream fps_ss;
    fps_ss << std::fixed << std::setprecision(1) << "Display FPS: " << m_current_fps.load();
    std::string fps_text = fps_ss.str();
    
    std::stringstream render_ss;
    render_ss << std::fixed << std::setprecision(2) << "Render time: " << m_last_render_time << " ms";
    std::string render_text = render_ss.str();
    
    // Draw background rectangle for better readability
    cv::Rect bg_rect(10, frame.rows - 80, 300, 70);
    cv::rectangle(frame, bg_rect, cv::Scalar(0, 0, 0), -1);
    
    // Draw text with information
    cv::putText(frame, fps_text, cv::Point(20, frame.rows - 50), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(frame, render_text, cv::Point(20, frame.rows - 20), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
}

void Display::updateFPS() {
    auto current_time = std::chrono::high_resolution_clock::now();
    auto time_since_last = std::chrono::duration<double, std::milli>(current_time - m_last_frame_time).count();
    
    // Update FPS if we have a valid time difference
    if (time_since_last > 0) {
        double instantaneous_fps = 1000.0 / time_since_last;
        
        // Apply simple low-pass filter for smoother FPS display (70% current, 30% new)
        double current = m_current_fps.load();
        double filtered = (current * 0.7) + (instantaneous_fps * 0.3);
        m_current_fps.store(filtered);
    }
    
    m_last_frame_time = current_time;
}

void Display::limitFrameRate() {
    if (m_max_fps <= 0 || !m_vsync_enabled) {
        return; // No limit or VSync disabled
    }
    
    auto now = std::chrono::high_resolution_clock::now();
    
    if (now < m_next_frame_time) {
        // Need to wait until next frame time
        std::this_thread::sleep_until(m_next_frame_time);
    }
    
    // Schedule next frame
    m_next_frame_time = std::chrono::high_resolution_clock::now() + m_frame_interval;
}