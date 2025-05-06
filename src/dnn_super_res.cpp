#include "dnn_super_res.h"

DnnSuperRes::DnnSuperRes(const std::string& model_path, 
                        const std::string& model_name, 
                        int scale,
                        ModelType type)
    : m_model_path(model_path),
    m_model_name(model_name),
    m_scale(scale),
    m_initialized(false),
    m_use_gpu(true),
    m_target_width(0),
    m_target_height(0),
    m_model_type(type) {
}

bool DnnSuperRes::initialize() {
    try {
        if (m_model_type == REAL_ESRGAN) {
            // Use EDSR model instead of trying to load RealESRGAN
            m_sr.readModel(m_model_path);
            m_sr.setModel(m_model_name, m_scale);
            
            if (m_use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
                std::cout << "Using CUDA backend for EDSR super-resolution" << std::endl;
                m_sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                m_sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            } else {
                std::cout << "Using CPU backend for EDSR super-resolution" << std::endl;
                m_sr.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                m_sr.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        } else {
            // Original code for other model types
            m_sr.readModel(m_model_path);
            m_sr.setModel(m_model_name, m_scale);
            
            if (m_use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
                std::cout << "Using CUDA backend for super-resolution" << std::endl;
                m_sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                m_sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            }
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
        
        // Use the appropriate model for upscaling - treat REAL_ESRGAN same as EDSR
        if (m_model_type == REAL_ESRGAN || m_model_type == EDSR) {
            // Use the OpenCV DNN Super Resolution implementation
            m_sr.upsample(input, output);
        } else {
            // Original upscaling code for other models
            m_sr.upsample(input, output);
        }
        
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

bool DnnSuperRes::upscaleRealESRGAN(const cv::Mat& input, cv::Mat& output) {
    try {
        // Pre-process input for RealESRGAN
        cv::Mat processed;
        preProcessRealESRGAN(input, processed);
        
        // Forward pass through the network
        m_net.setInput(processed);
        cv::Mat result = m_net.forward();
        
        // Post-process the result
        postProcessRealESRGAN(result, output);
        
        // Apply unified pixel enhancement
        cv::Mat enhanced;
        
        // First, create a blurred version that will unify the flat areas
        cv::Mat blurred;
        cv::bilateralFilter(output, blurred, 5, 15, 15); // Preserve edges, smooth flat areas
        
        // Create an edge mask to identify detailed areas
        cv::Mat gray, edges, edge_mask;
        cv::cvtColor(output, gray, cv::COLOR_BGR2GRAY);
        cv::Canny(gray, edges, 50, 150);
        cv::dilate(edges, edge_mask, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3,3)));
        
        // Smooth the mask
        cv::GaussianBlur(edge_mask, edge_mask, cv::Size(5,5), 0);
        
        // Convert mask to floating point 0-1 range
        edge_mask.convertTo(edge_mask, CV_32F, 1.0/255.0);
        
        // Blend original with smoothed version based on edge mask
        // Keep details at edges, smooth flat areas for unified appearance
        output.convertTo(enhanced, CV_32F);
        blurred.convertTo(blurred, CV_32F);
        
        cv::Mat result_float;
        for (int i = 0; i < enhanced.rows; i++) {
            for (int j = 0; j < enhanced.cols; j++) {
                float mask_value = edge_mask.at<float>(i, j);
                enhanced.at<cv::Vec3f>(i, j) = 
                    enhanced.at<cv::Vec3f>(i, j) * mask_value + 
                    blurred.at<cv::Vec3f>(i, j) * (1.0f - mask_value);
            }
        }
        
        enhanced.convertTo(output, CV_8U);
        
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error in RealESRGAN upscaling: " << e.what() << std::endl;
        
        // Fallback to standard resizing
        cv::resize(input, output, cv::Size(input.cols * m_scale, input.rows * m_scale), 
                  0, 0, cv::INTER_LANCZOS4);
        return true;
    }
}


void DnnSuperRes::preProcessRealESRGAN(const cv::Mat& input, cv::Mat& processed) {
    // Convert BGR to RGB (RealESRGAN expects RGB)
    cv::Mat rgb;
    cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
    
    // Convert to float32 and normalize to 0-1
    cv::Mat floatInput;
    rgb.convertTo(floatInput, CV_32F, 1.0/255.0);
    
    // RealESRGAN expects NCHW format (batch, channels, height, width)
    processed = cv::dnn::blobFromImage(floatInput, 1.0, cv::Size(), 
                                     cv::Scalar(0, 0, 0), true, false);
}

void DnnSuperRes::postProcessRealESRGAN(const cv::Mat& processed, cv::Mat& output) {
    // Get dimensions
    int height = processed.size[2];
    int width = processed.size[3];
    
    // Extract channels
    std::vector<cv::Mat> channels;
    for (int i = 0; i < 3; i++) {
        cv::Mat channel(height, width, CV_32F, 
                      const_cast<float*>(processed.ptr<float>(0, i)));
        channels.push_back(channel);
    }
    
    // Merge channels
    cv::Mat result;
    cv::merge(channels, result);
    
    // Convert back to 0-255 range
    result = result * 255.0;
    
    // Convert to 8-bit
    result.convertTo(result, CV_8U);
    
    // Convert RGB back to BGR
    cv::cvtColor(result, output, cv::COLOR_RGB2BGR);
}
