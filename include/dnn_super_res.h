#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn_superres.hpp>  // Add this include
#include <string>
#include <memory>
#include <iostream>

class DnnSuperRes {
public:
    // Constructor - specify model path and scale factor
    DnnSuperRes(const std::string& model_path = "models/FSRCNN_x4.pb", 
                const std::string& model_name = "fsrcnn", 
                int scale = 4);
    
    // Initialize the model
    bool initialize();
    
    // Upscale an image
    bool upscale(const cv::Mat& input, cv::Mat& output);
    
    // Check if model is loaded
    bool isInitialized() const { return m_initialized; }
    
    // Set target dimensions for output
    void setTargetSize(int width, int height) {
        m_target_width = width;
        m_target_height = height;
    }
    
    // Set to use GPU if available
    void setUseGPU(bool use_gpu) { m_use_gpu = use_gpu; }
    
private:
    std::string m_model_path;
    std::string m_model_name;
    int m_scale;
    bool m_initialized;
    bool m_use_gpu;
    cv::dnn::Net m_net;
    int m_target_width;
    int m_target_height;
    
    // Add this line to fix the compilation error
    cv::dnn_superres::DnnSuperResImpl m_sr;
};