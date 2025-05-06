#include "dnn_super_res.h"

DnnSuperRes::DnnSuperRes(const std::string& model_path, 
                         const std::string& model_name, 
                         int scale)
    : m_model_path(model_path),
      m_model_name(model_name),
      m_scale(scale),
      m_initialized(false),
      m_use_gpu(true),
      m_target_width(0),
      m_target_height(0) {
}

bool DnnSuperRes::initialize() {
    try {
        // Read the model
        m_sr.readModel(m_model_path);
        
        // Set the model and scale
        m_sr.setModel(m_model_name, m_scale);
        
        // Set the backend and target
        if (m_use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
            std::cout << "Using CUDA backend for super-resolution" << std::endl;
            m_sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            m_sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        }
        
        m_initialized = true;
        std::cout << "Super-resolution model loaded successfully" << std::endl;
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Failed to initialize SR: " << e.what() << std::endl;
        m_initialized = false;
        return false;
    }
}


bool DnnSuperRes::upscale(const cv::Mat& input, cv::Mat& output) {
    if (!m_initialized) {
        std::cerr << "Super-resolution model not initialized" << std::endl;
        return false;
    }
    
    if (input.empty()) {
        std::cerr << "Input image is empty" << std::endl;
        return false;
    }
    
    try {
        // Start timing
        auto start = std::chrono::high_resolution_clock::now();
        
        // Upscale using the dnn_superres module
        m_sr.upsample(input, output);
        
        // Resize to target dimensions if specified
        if (m_target_width > 0 && m_target_height > 0 && 
            (output.cols != m_target_width || output.rows != m_target_height)) {
            cv::resize(output, output, cv::Size(m_target_width, m_target_height), 
                      0, 0, cv::INTER_LANCZOS4);
        }
        
        // Calculate and print processing time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Super-resolution processing time: " << duration.count() << " ms" << std::endl;
        
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error in DNN super-resolution: " << e.what() << std::endl;
        
        // Fall back to Lanczos resize
        if (m_target_width > 0 && m_target_height > 0) {
            cv::resize(input, output, cv::Size(m_target_width, m_target_height), 
                      0, 0, cv::INTER_LANCZOS4);
            return true;
        }
        return false;
    }
}