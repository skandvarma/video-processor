#include "camera.h"
#include <iostream>
#include <thread>
#include <chrono>

Camera::Camera(int camera_index) 
    : width(0), height(0), fps(0), initialized(false), camera_index(camera_index), is_file(false),
      has_new_frame(false), thread_running(false) {
}

Camera::Camera(const std::string& video_source) 
    : width(0), height(0), fps(0), initialized(false), camera_index(-1), 
      video_source(video_source), is_file(true), has_new_frame(false), thread_running(false) {
}

Camera::~Camera() {
    // Stop the grabbing thread if it's running
    thread_running = false;
    if (grab_thread.joinable()) {
        grab_thread.join();
    }
    
    // Release the capture device
    if (cap && cap->isOpened()) {
        cap->release();
    }
}

std::vector<int> Camera::listAvailableCameras() {
    std::vector<int> availableCameras;
    
    for (int i = 0; i < 10; i++) {
        cv::VideoCapture temp(i);
        if (temp.isOpened()) {
            std::cout << "Camera " << i << " is available" << std::endl;
            availableCameras.push_back(i);
            temp.release();
        }
    }
    
    return availableCameras;
}

bool Camera::initialize(int w, int h, int framerate) {
    // If this is a video file, use a different approach
    if (is_file) {
        // Try to open with FFMPEG backend directly for video files
        cap = std::make_unique<cv::VideoCapture>(video_source, cv::CAP_FFMPEG);
        
        if (cap->isOpened()) {
            // Get the actual properties
            width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
            height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
            fps = cap->get(cv::CAP_PROP_FPS);
            
            std::cout << "Successfully opened video file: " << video_source << std::endl;
            std::cout << "Video properties: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
            
            // Try to get a test frame to confirm everything is working
            cv::Mat test_frame;
            bool success = cap->read(test_frame);
            
            if (success && !test_frame.empty()) {
                // Reset the video to the beginning
                cap->set(cv::CAP_PROP_POS_FRAMES, 0);
                
                initialized = true;
                return true;
            } else {
                std::cerr << "Could open the file but failed to read a frame" << std::endl;
                cap->release();
                return false;
            }
        } else {
            std::cerr << "Failed to open video file: " << video_source << std::endl;
            
            // Try with other backends as fallback
            std::vector<int> backends = {
                cv::CAP_GSTREAMER,
                cv::CAP_ANY
            };
            
            for (auto backend : backends) {
                std::cout << "Trying backend " << backend << " for video file" << std::endl;
                cap = std::make_unique<cv::VideoCapture>(video_source, backend);
                
                if (cap->isOpened()) {
                    width = cap->get(cv::CAP_PROP_FRAME_WIDTH);
                    height = cap->get(cv::CAP_PROP_FRAME_HEIGHT);
                    fps = cap->get(cv::CAP_PROP_FPS);
                    
                    std::cout << "Successfully opened video file with backend " << backend << std::endl;
                    std::cout << "Video properties: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
                    
                    initialized = true;
                    return true;
                }
            }
            
            return false;
        }
    }
    
    // For camera sources, proceed with the original method
    // First, try to query the camera's supported formats
    bool found_suitable_format = false;
    std::string format_cmd = "v4l2-ctl --device=/dev/video" + std::to_string(camera_index) + " --list-formats-ext";
    bool supports_mjpg = system((format_cmd + " | grep -q MJPG").c_str()) == 0;
    bool supports_h264 = system((format_cmd + " | grep -q H264").c_str()) == 0;
    bool supports_yuyv = system((format_cmd + " | grep -q YUYV").c_str()) == 0;
    
    std::cout << "Camera format support - MJPG: " << supports_mjpg 
              << ", H264: " << supports_h264 
              << ", YUYV: " << supports_yuyv << std::endl;
    
    // Try the formats in order of preference and capability
    std::vector<std::pair<std::string, std::string>> pipeline_configs;
    
    // 1. Try MJPG format if supported
    if (supports_mjpg) {
        pipeline_configs.push_back({
            "MJPG format",
            "v4l2src device=/dev/video" + std::to_string(camera_index) + 
            " ! image/jpeg,width=" + std::to_string(w) + 
            ",height=" + std::to_string(h) + 
            ",framerate=" + std::to_string(framerate) + "/1" +
            " ! jpegdec ! videoconvert ! appsink"
        });
    }
    
    // 2. Try H264 format if supported
    if (supports_h264) {
        pipeline_configs.push_back({
            "H264 format",
            "v4l2src device=/dev/video" + std::to_string(camera_index) + 
            " ! video/x-h264,width=" + std::to_string(w) + 
            ",height=" + std::to_string(h) + 
            ",framerate=" + std::to_string(framerate) + "/1" +
            " ! h264parse ! avdec_h264 ! videoconvert ! appsink"
        });
    }
    
    // 3. Try YUYV/YUY2 format if supported
    if (supports_yuyv) {
        pipeline_configs.push_back({
            "YUYV format",
            "v4l2src device=/dev/video" + std::to_string(camera_index) + 
            " ! video/x-raw,format=YUY2,width=" + std::to_string(w) + 
            ",height=" + std::to_string(h) + 
            ",framerate=" + std::to_string(framerate) + "/1" +
            " ! videoconvert ! appsink"
        });
    }
    
    // 4. Always add a generic x-raw fallback
    pipeline_configs.push_back({
        "Generic raw format",
        "v4l2src device=/dev/video" + std::to_string(camera_index) + 
        " ! video/x-raw,width=" + std::to_string(w) + 
        ",height=" + std::to_string(h) + 
        " ! videoconvert ! appsink"
    });
    
    // 5. Add an optimized raw format for better performance
    pipeline_configs.push_back({
        "Optimized raw format",
        "v4l2src device=/dev/video" + std::to_string(camera_index) + 
        " ! video/x-raw,width=" + std::to_string(w) + 
        ",height=" + std::to_string(h) + 
        " ! queue max-size-buffers=5 leaky=downstream ! videoconvert ! video/x-raw,format=BGR ! appsink drop=true"
    });
    
    // 6. Final fallback with minimal constraints
    pipeline_configs.push_back({
        "Minimal constraints",
        "v4l2src device=/dev/video" + std::to_string(camera_index) + 
        " ! videoconvert ! appsink"
    });
    
    // Try each pipeline configuration
    for (const auto& [name, pipeline_str] : pipeline_configs) {
        std::cout << "Trying pipeline: " << name << std::endl;
        std::cout << pipeline_str << std::endl;
        
        cap = std::make_unique<cv::VideoCapture>(pipeline_str, cv::CAP_GSTREAMER);
        
        if (cap->isOpened()) {
            // Try to get a test frame
            cv::Mat test_frame;
            bool success = cap->read(test_frame);
            
            if (success && !test_frame.empty()) {
                std::cout << "Successfully opened camera with " << name << std::endl;
                std::cout << "Frame size: " << test_frame.cols << "x" << test_frame.rows << std::endl;
                
                // Store actual dimensions
                width = test_frame.cols;
                height = test_frame.rows;
                fps = framerate; // Assume requested framerate
                
                initialized = true;
                return true;
            } else {
                std::cout << "Pipeline opened but failed to grab frame" << std::endl;
                cap->release();
            }
        } else {
            std::cout << "Failed to open pipeline" << std::endl;
        }
    }
    
    // If all GStreamer methods fail, try standard OpenCV backends
    std::vector<int> backends = {
        cv::CAP_V4L2,
        cv::CAP_V4L,
        cv::CAP_FFMPEG,
        cv::CAP_ANY
    };
    
    for (auto backend : backends) {
        std::cout << "Trying OpenCV backend: " << backend << std::endl;
        cap = std::make_unique<cv::VideoCapture>(camera_index, backend);
        
        if (cap->isOpened()) {
            // Try to set properties
            cap->set(cv::CAP_PROP_FRAME_WIDTH, w);
            cap->set(cv::CAP_PROP_FRAME_HEIGHT, h);
            cap->set(cv::CAP_PROP_FPS, framerate);
            
            // Try to get a test frame
            cv::Mat test_frame;
            bool success = cap->read(test_frame);
            
            if (success && !test_frame.empty()) {
                width = test_frame.cols;
                height = test_frame.rows;
                fps = cap->get(cv::CAP_PROP_FPS);
                
                std::cout << "Successfully opened camera with backend " << backend << std::endl;
                std::cout << "Actual properties: " << width << "x" << height << " @ " << fps << " FPS" << std::endl;
                
                initialized = true;
                return true;
            } else {
                cap->release();
            }
        }
    }
    
    std::cerr << "Failed to initialize camera with any method" << std::endl;
    return false;
}

void Camera::grabLoop() {
    cv::Mat frame;
    int frames_grabbed = 0;
    int frames_dropped = 0;
    
    // For measuring real grab rate
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (thread_running) {
        if (cap && cap->isOpened()) {
            // Use grab() which is faster than read()
            bool grabbed = cap->grab();
            
            if (grabbed) {
                frames_grabbed++;
                
                // Only retrieve if we can store it (non-blocking check)
                std::unique_lock<std::mutex> lock(frame_mutex, std::try_to_lock);
                if (lock.owns_lock()) {
                    if (cap->retrieve(frame)) {
                        if (!frame.empty()) {
                            frame.copyTo(latest_frame);
                            has_new_frame = true;
                        }
                    }
                } else {
                    // Previous frame hasn't been consumed yet
                    frames_dropped++;
                }
                
                // Print grab rate occasionally
                if (frames_grabbed % 100 == 0) {
                    auto current_time = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration<double>(current_time - start_time).count();
                    double grab_rate = frames_grabbed / elapsed;
                    
                    std::cout << "Background thread: grabbing at " << grab_rate 
                              << " FPS, dropped " << frames_dropped << " frames" << std::endl;
                    
                    // Reset counters
                    frames_grabbed = 0;
                    frames_dropped = 0;
                    start_time = current_time;
                }
            }
        }
        
        // Sleep just enough to avoid burning CPU
        // Adjust this value if needed - smaller values might give higher FPS
        // but will use more CPU
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    
    std::cout << "Background grab thread exiting" << std::endl;
}

bool Camera::getFrame(cv::Mat& frame) {
    if (!initialized || !cap || !cap->isOpened()) {
        return false;
    }
    
    if (has_new_frame) {
        std::lock_guard<std::mutex> lock(frame_mutex);
        if (!latest_frame.empty()) {
            latest_frame.copyTo(frame);
            has_new_frame = false;
            return true;
        }
    }
    
    // Fallback to direct capture if no frame available
    return cap->read(frame);
}

bool Camera::isOpened() const {
    return (cap && cap->isOpened());
}

double Camera::getFPS() const {
    return fps;
}

int Camera::getWidth() const {
    return width;
}

int Camera::getHeight() const {
    return height;
}

bool Camera::tryBackends() {
    // This method could be implemented if needed
    return false;
}