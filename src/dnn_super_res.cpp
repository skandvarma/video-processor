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
            // Load ONNX model
            std::cout << "Loading ONNX model: " << m_model_path << std::endl;
            m_net = cv::dnn::readNetFromONNX(m_model_path);
            
            if (m_net.empty()) {
                std::cerr << "Failed to load ONNX model: " << m_model_path << std::endl;
                m_initialized = false;
                return false;
            }
            
            if (m_use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
                std::cout << "Using CUDA backend for ONNX super-resolution" << std::endl;
                m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            } else {
                std::cout << "Using CPU backend for ONNX super-resolution" << std::endl;
                m_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                m_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
            
            // Very important - explicitly set initialized flag
            m_initialized = true;
        } else {
            // Original code for other model types
            m_sr.readModel(m_model_path);
            m_sr.setModel(m_model_name, m_scale);
            
            if (m_use_gpu && cv::cuda::getCudaEnabledDeviceCount() > 0) {
                std::cout << "Using CUDA backend for super-resolution" << std::endl;
                m_sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                m_sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            }
            
            m_initialized = true;
        }
        
        std::cout << "Super-resolution model loaded successfully" << std::endl;
        return m_initialized;
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
        
        // Use the appropriate model for upscaling
        if (m_model_type == REAL_ESRGAN) {
            // Use ONNX-based upscaling for RealESRGAN
            return upscaleRealESRGAN(input, output);
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
        // Print input size for debugging
        std::cout << "Input size: " << input.cols << "x" << input.rows << " channels: " << input.channels() << std::endl;
        
        // 1. Pre-process: Convert BGR to RGB and normalize
        cv::Mat rgb;
        cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
        
        // Convert to float and normalize to 0-1 range
        cv::Mat floatImg;
        rgb.convertTo(floatImg, CV_32F, 1.0/255.0);
        
        // Create blob in NCHW format (batch, channels, height, width)
        cv::Mat inputBlob = cv::dnn::blobFromImage(floatImg);
        std::cout << "Input blob shape: ";
        for (int i = 0; i < inputBlob.dims; i++) {
            std::cout << inputBlob.size[i] << " ";
        }
        std::cout << std::endl;
        
        // 2. Run inference
        m_net.setInput(inputBlob);
        cv::Mat outBlob = m_net.forward();
        
        // 3. Post-process: Convert back to image format
        std::cout << "Output blob shape: ";
        for (int i = 0; i < outBlob.dims; i++) {
            std::cout << outBlob.size[i] << " ";
        }
        std::cout << std::endl;
        
        // Extract dimensions - expected 4D: [1, 3, H*4, W*4]
        if (outBlob.dims != 4) {
            std::cerr << "Unexpected model output format" << std::endl;
            return false;
        }
        
        int channels = outBlob.size[1];
        int height = outBlob.size[2];
        int width = outBlob.size[3];
        
        // Create a Mat object from the blob
        std::vector<cv::Mat> outputChannels;
        
        // For each channel in the output
        for (int c = 0; c < channels; c++) {
            // Get pointer to this channel's data
            cv::Mat channel(height, width, CV_32F, (float*)outBlob.ptr(0, c));
            outputChannels.push_back(channel);
        }
        
        // Merge the channels back into a color image
        cv::Mat processedRgb;
        cv::merge(outputChannels, processedRgb);
        
        // Convert back to 0-255 range and to 8-bit
        processedRgb = processedRgb * 255.0;
        processedRgb.convertTo(processedRgb, CV_8U);
        
        // Convert back to BGR
        cv::cvtColor(processedRgb, output, cv::COLOR_RGB2BGR);
        
        std::cout << "Final output size: " << output.cols << "x" << output.rows << std::endl;
        
        // Resize to target dimensions if needed
        if (m_target_width > 0 && m_target_height > 0 && 
            (output.cols != m_target_width || output.rows != m_target_height)) {
            cv::resize(output, output, cv::Size(m_target_width, m_target_height), 
                       0, 0, cv::INTER_LANCZOS4);
        }
        
        return true;
    }
    catch (const cv::Exception& e) {
        std::cerr << "Error in RealESRGAN upscaling: " << e.what() << std::endl;
        
        // Create a visible error image
        output = cv::Mat(input.rows * 4, input.cols * 4, CV_8UC3, cv::Scalar(255, 0, 0));
        
        // Try falling back to standard resize - this should work
        try {
            cv::resize(input, output, cv::Size(input.cols * 4, input.rows * 4), 
                      0, 0, cv::INTER_LANCZOS4);
                      
            // Add error text to the image
            cv::putText(output, "ESRGAN Error - Using standard resize", 
                      cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                      cv::Scalar(0, 0, 255), 2);
        } 
        catch (...) {
            // Last resort
            std::cerr << "Even fallback resize failed!" << std::endl;
        }
        
        return true; // Return true so processing continues with fallback image
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
