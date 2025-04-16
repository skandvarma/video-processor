#include "pipeline.h"
#include "camera.h"
#include "frame_buffer.h"
#include "upscaler.h"
#include "display.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

// Helper define for time measurement
using Clock = std::chrono::high_resolution_clock;

// Implementation class (PIMPL pattern)
class Pipeline::Impl {
public:
    Impl(const Pipeline::Config& config, std::atomic<double>& latency, std::atomic<double>& fps, Timer& timer) 
        : m_config(config),
          m_running(false),
          m_frame_counter(0),
          m_latency_ref(latency),
          m_fps_ref(fps),
          m_timer_ref(timer) {
    }
    
    ~Impl() {
        // Make sure to stop the pipeline
        if (m_running.load()) {
            this->stop();
        }
    }
    
    bool initialize(int camera_index) {
        // Update camera index if provided
        if (camera_index >= 0) {
            m_config.camera_index = camera_index;
        }
        
        // Initialize camera - improved with direct access to camera index
        try {
            // First verify the camera index is valid
            auto available_cameras = Camera::listAvailableCameras();
            
            if (available_cameras.empty()) {
                std::cerr << "No cameras detected!" << std::endl;
                return false;
            }
            
            // Make sure we use a valid camera index
            if (std::find(available_cameras.begin(), available_cameras.end(), m_config.camera_index) 
                == available_cameras.end()) {
                std::cout << "Camera index " << m_config.camera_index << " not available." << std::endl;
                m_config.camera_index = available_cameras[0];
                std::cout << "Using camera index " << m_config.camera_index << " instead." << std::endl;
            }
            
            // Now create and initialize the camera with the validated index
            m_camera = std::make_unique<Camera>(m_config.camera_index);
            if (!m_camera->initialize(m_config.camera_width, m_config.camera_height, m_config.camera_fps)) {
                std::cerr << "Failed to initialize camera with index " << m_config.camera_index << std::endl;
                return false;
            }
            
            std::cout << "Camera initialized successfully at " 
                      << m_camera->getWidth() << "x" << m_camera->getHeight() 
                      << " @ " << m_camera->getFPS() << " FPS" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error creating camera: " << e.what() << std::endl;
            return false;
        }
        
        // Initialize upscaler
        try {
            m_upscaler = std::make_unique<Upscaler>(m_config.upscale_algorithm, m_config.use_gpu);
            if (!m_upscaler->initialize(m_config.target_width, m_config.target_height)) {
                std::cerr << "Failed to initialize upscaler" << std::endl;
                return false;
            }
            
            std::cout << "Upscaler initialized with algorithm: " << m_upscaler->getAlgorithmName()
                     << ", using " << (m_upscaler->isUsingGPU() ? "GPU" : "CPU") << std::endl; 
        } catch (const std::exception& e) {
            std::cerr << "Error creating upscaler: " << e.what() << std::endl;
            return false;
        }
        
        // Initialize display
        try {
            m_display = std::make_unique<Display>(m_config.target_width, m_config.target_height);
            if (!m_display->initialize(m_config.window_name)) {
                std::cerr << "Failed to initialize display" << std::endl;
                return false;
            }
            
            // Configure display options
            m_display->showPerformanceMetrics(m_config.show_metrics);
            m_display->setVSync(m_config.enable_vsync);
            m_display->setMaxFrameRate(m_config.max_display_fps);
            std::cout << "Display initialized at " << m_config.target_width << "x" 
                     << m_config.target_height << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error creating display: " << e.what() << std::endl;
            return false;
        }
        
        // Initialize frame buffer
        try {
            m_buffer = std::make_unique<FrameBuffer>(m_config.buffer_size);
            std::cout << "Frame buffer initialized with size " << m_config.buffer_size << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error creating frame buffer: " << e.what() << std::endl;
            return false;
        }
        
        // Reset performance metrics
        m_timer_ref.reset();
        m_latency_ref.store(0.0);
        m_fps_ref.store(0.0);
        m_frame_counter = 0;
        m_last_fps_update = Clock::now();
        
        std::cout << "Pipeline initialized successfully" << std::endl;
        return true;
    }
    
    bool start() {
        // Check if components are initialized
        if (!m_camera || !m_upscaler || !m_display || !m_buffer) {
            std::cerr << "Cannot start: Pipeline not fully initialized" << std::endl;
            return false;
        }
        
        // Check if already running
        if (m_running.load()) {
            std::cout << "Pipeline is already running" << std::endl;
            return true;
        }
        
        // Verify camera is opened
        if (!m_camera->isOpened()) {
            std::cerr << "Camera is not opened, cannot start pipeline" << std::endl;
            return false;
        }
        
        // Set running flag
        m_running.store(true);
        
        // Start threads
        try {
            m_capture_thread = std::make_unique<std::thread>(&Pipeline::Impl::captureLoop, this);
            m_processing_thread = std::make_unique<std::thread>(&Pipeline::Impl::processingLoop, this);
            m_display_thread = std::make_unique<std::thread>(&Pipeline::Impl::displayLoop, this);
            
            std::cout << "Pipeline started with " << m_config.buffer_size 
                    << " frame buffer" << std::endl;
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Failed to start pipeline threads: " << e.what() << std::endl;
            this->stop();
            return false;
        }
    }
    
    void stop() {
        // Signal threads to stop
        m_running.store(false);
        
        // Notify any waiting threads
        m_pipeline_cv.notify_all();
        
        // Join threads if they're running
        if (m_capture_thread && m_capture_thread->joinable()) {
            m_capture_thread->join();
            m_capture_thread.reset();
        }
        
        if (m_processing_thread && m_processing_thread->joinable()) {
            m_processing_thread->join();
            m_processing_thread.reset();
        }
        
        if (m_display_thread && m_display_thread->joinable()) {
            m_display_thread->join();
            m_display_thread.reset();
        }
        
        // Clear the buffer
        if (m_buffer) {
            m_buffer->clear();
        }
        
        std::cout << "Pipeline stopped" << std::endl;
    }
    
    bool isRunning() const {
        return m_running.load();
    }
    
    bool waitForKey(int key) {
        while (isRunning()) {
            int pressed = cv::waitKey(100); // Check every 100ms
            
            if (pressed == key) {
                stop();
                return true;
            }
            else if (pressed >= 0) {
                // Some other key was pressed
                return false;
            }
        }
        
        return false;
    }
    
    void setDisplayOptions(bool show_metrics) {
        if (m_display) {
            m_display->showPerformanceMetrics(show_metrics);
        }
    }
    
    void printPerformanceStats() const {
        std::cout << "\n=== Pipeline Performance ===" << std::endl;
        std::cout << "End-to-end latency: " << std::fixed << std::setprecision(2) 
                << m_latency_ref.load() << " ms" << std::endl;
        std::cout << "Effective FPS: " << std::fixed << std::setprecision(1) 
                << m_fps_ref.load() << std::endl;
        
        // Print detailed component timing from the timer
        m_timer_ref.printStats();
    }
    
private:
    // Components (using unique_ptr to avoid copy/move issues)
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<FrameBuffer> m_buffer;
    std::unique_ptr<Upscaler> m_upscaler;
    std::unique_ptr<Display> m_display;
    
    // Configuration
    Pipeline::Config m_config;
    
    // Threading
    std::unique_ptr<std::thread> m_capture_thread;
    std::unique_ptr<std::thread> m_processing_thread;
    std::unique_ptr<std::thread> m_display_thread;
    std::atomic<bool> m_running;
    
    // Synchronization
    std::condition_variable m_pipeline_cv;
    std::mutex m_pipeline_mutex;
    
    // Performance monitoring (references to outer class members)
    std::atomic<double>& m_latency_ref;
    std::atomic<double>& m_fps_ref;
    Timer& m_timer_ref;
    
    // Local performance tracking
    std::chrono::time_point<std::chrono::high_resolution_clock> m_last_fps_update;
    int m_frame_counter;
    std::mutex m_perf_mutex;
    
    // Thread loop functions
    void captureLoop() {
        std::cout << "Capture thread started" << std::endl;
        cv::Mat frame;
        int dropped_frames = 0;
        
        while (m_running.load()) {
            // Measure capture time
            m_timer_ref.start("capture");
            auto capture_start_time = Clock::now();
            
            // Get frame from camera
            bool success = m_camera->getFrame(frame);
            
            if (!success || frame.empty()) {
                std::cerr << "Failed to capture frame" << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            m_timer_ref.stop("capture");
            
            // Try to push to buffer (non-blocking)
            m_timer_ref.start("buffer_push");
            bool pushed = m_buffer->pushFrame(frame, false);
            m_timer_ref.stop("buffer_push");
            
            if (!pushed) {
                // Buffer full, frame dropped
                dropped_frames++;
                
                if (dropped_frames % 10 == 0) {
                    std::cerr << "Warning: Dropped " << dropped_frames << " frames due to full buffer" << std::endl;
                }
            }
            
            // Update frame counter for FPS calculation
            {
                std::lock_guard<std::mutex> lock(m_perf_mutex);
                m_frame_counter++;
                
                // Update FPS every second
                auto now = Clock::now();
                auto elapsed = std::chrono::duration<double>(now - m_last_fps_update).count();
                
                if (elapsed >= 1.0) {
                    m_fps_ref.store(m_frame_counter / elapsed);
                    m_frame_counter = 0;
                    m_last_fps_update = now;
                }
            }
        }
        
        std::cout << "Capture thread exiting" << std::endl;
    }
    
    void processingLoop() {
        std::cout << "Processing thread started" << std::endl;
        cv::Mat input_frame, output_frame;
        
        // Use a structure to store frame timestamps for latency tracking
        struct FrameData {
            cv::Mat frame;
            std::chrono::time_point<std::chrono::high_resolution_clock> capture_time;
        };
        
        while (m_running.load()) {
            // Get frame from buffer (blocking)
            m_timer_ref.start("buffer_pop");
            bool success = m_buffer->popFrame(input_frame, true);
            m_timer_ref.stop("buffer_pop");
            
            if (!success || input_frame.empty()) {
                // Buffer might be empty due to shutdown
                if (!m_running.load()) {
                    break;
                }
                
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            
            // Record when we got this frame for processing
            auto process_time = Clock::now();
            
            // Upscale the frame
            m_timer_ref.start("upscale");
            bool upscale_success = m_upscaler->upscale(input_frame, output_frame);
            m_timer_ref.stop("upscale");
            
            if (!upscale_success) {
                std::cerr << "Failed to upscale frame" << std::endl;
                continue;
            }
            
            // Pass the processed frame to the display thread via another buffer
            FrameData fd{output_frame, process_time};
            
            // In a real implementation, we would pass this directly to the display thread
            // with frame metadata. For this example, we'll just render immediately.
            m_display->renderFrame(output_frame);
            
            // Measure latency from capture to display
            auto display_time = Clock::now();
            auto latency = std::chrono::duration<double, std::milli>(display_time - process_time).count();
            
            // Update moving average of latency (simple low-pass filter)
            double current = m_latency_ref.load();
            double updated = (current * 0.9) + (latency * 0.1); // 90% old, 10% new
            m_latency_ref.store(updated);
        }
        
        std::cout << "Processing thread exiting" << std::endl;
    }
    
    void displayLoop() {
        std::cout << "Display thread started" << std::endl;
        
        // This implementation uses direct rendering in the processing thread
        // In a more advanced implementation, this would be a separate thread
        // that takes frames from a queue and renders them with precise timing
        
        // For now, just keep the thread alive
        while (m_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        
        std::cout << "Display thread exiting" << std::endl;
    }
};

// Pipeline implementation (delegates to Impl)
Pipeline::Pipeline() 
    : m_impl(std::make_unique<Impl>(Config(), m_latency, m_fps, m_timer)) {
}

Pipeline::Pipeline(const Config& config) 
    : m_config(config),
      m_impl(std::make_unique<Impl>(config, m_latency, m_fps, m_timer)) {
}

Pipeline::~Pipeline() = default;

bool Pipeline::initialize(int camera_index) {
    return m_impl->initialize(camera_index);
}

bool Pipeline::start() {
    return m_impl->start();
}

void Pipeline::stop() {
    m_impl->stop();
}

bool Pipeline::isRunning() const {
    return m_impl->isRunning();
}

bool Pipeline::waitForKey(int key) {
    return m_impl->waitForKey(key);
}

double Pipeline::getLatency() const {
    return m_latency.load();
}

double Pipeline::getFPS() const {
    return m_fps.load();
}

void Pipeline::setTargetResolution(int width, int height) {
    if (isRunning()) {
        std::cerr << "Cannot change resolution while pipeline is running" << std::endl;
        return;
    }
    
    m_config.target_width = width;
    m_config.target_height = height;
}

void Pipeline::setBufferSize(int size) {
    if (isRunning()) {
        std::cerr << "Cannot change buffer size while pipeline is running" << std::endl;
        return;
    }
    
    m_config.buffer_size = size;
}

void Pipeline::setDisplayOptions(bool show_metrics) {
    m_config.show_metrics = show_metrics;
    m_impl->setDisplayOptions(show_metrics);
}

void Pipeline::printPerformanceStats() const {
    m_impl->printPerformanceStats();
}