#include "selective_bilateral.h"
#include <iostream>
#include <algorithm>

// Check if OpenCV was built with CUDA support
#ifdef WITH_CUDA
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

SelectiveBilateral::SelectiveBilateral() 
    : m_initialized(false) {
}

SelectiveBilateral::SelectiveBilateral(const Config& config)
    : m_config(config), m_initialized(false) {
}

SelectiveBilateral::~SelectiveBilateral() {
}

bool SelectiveBilateral::initialize() {
    // Check if GPU is available when requested
    if (m_config.use_gpu) {
#ifdef WITH_CUDA
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
            std::cout << "CUDA requested for bilateral filtering but not available. Using CPU fallback." << std::endl;
            m_config.use_gpu = false;
        } else {
            std::cout << "Using CUDA for bilateral filtering." << std::endl;
        }
#else
        std::cout << "CUDA requested for bilateral filtering but OpenCV was built without CUDA support. Using CPU fallback." << std::endl;
        m_config.use_gpu = false;
#endif
    }
    
    // If we're using multi-scale, adjust parameters for each scale
    if (m_config.use_multiscale) {
        // Ensures that the number of scales is reasonable
        m_config.num_scales = std::max(1, std::min(m_config.num_scales, 5));
    }
    
    m_initialized = true;
    return true;
}

void SelectiveBilateral::setConfig(const Config& config) {
    m_config = config;
}

SelectiveBilateral::Config SelectiveBilateral::getConfig() const {
    return m_config;
}

bool SelectiveBilateral::process(const cv::Mat& input, cv::Mat& output) {
    if (!m_initialized) {
        std::cerr << "Selective bilateral filtering module not initialized" << std::endl;
        return false;
    }
    
    if (input.empty()) {
        std::cerr << "Empty input image" << std::endl;
        return false;
    }
    
    try {
        // Adjust processing approach based on configuration
        if (m_config.use_multiscale) {
            return applyMultiscaleBilateral(input, output);
        } else if (m_config.selective) {
            return applySelectiveBilateral(input, output);
        } else {
            return applyBilateralFilter(input, output);
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error in selective bilateral filtering: " << e.what() << std::endl;
        input.copyTo(output);
        return false;
    }
}

bool SelectiveBilateral::applyBilateralFilter(const cv::Mat& input, cv::Mat& output) {
    try {
        // Adapt parameters based on input if configured
        int diameter = m_config.diameter;
        double sigma_color = m_config.sigma_color;
        double sigma_space = m_config.sigma_space;
        
        if (m_config.adaptive_params) {
            calculateAdaptiveParams(input, diameter, sigma_color, sigma_space);
        }
        
        // Apply bilateral filter with GPU if available and requested
        if (m_config.use_gpu) {
#ifdef WITH_CUDA
    cv::cuda::GpuMat d_input, d_output;
    d_input.upload(input);
    // CUDA bilateral via free function:
    cv::cuda::bilateralFilter(d_input, d_output,
                              diameter, sigma_color, sigma_space);
    d_output.download(output);
#else
    cv::bilateralFilter(input, output, diameter, sigma_color, sigma_space);
#endif

        } else {
            // CPU implementation
            cv::bilateralFilter(input, output, diameter, sigma_color, sigma_space);
        }
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error applying bilateral filter: " << e.what() << std::endl;
        input.copyTo(output);
        return false;
    }
}

bool SelectiveBilateral::applySelectiveBilateral(const cv::Mat& input, cv::Mat& output) {
    try {
        // Create detail mask for selective processing
        cv::Mat detail_mask;
        if (!createDetailMask(input, detail_mask)) {
            std::cerr << "Failed to create detail mask" << std::endl;
            return applyBilateralFilter(input, output); // Fallback to standard bilateral
        }
        
        // Apply joint bilateral filtering using the detail mask
        return applyJointBilateral(input, detail_mask, output);
    } catch (const cv::Exception& e) {
        std::cerr << "Error applying selective bilateral filter: " << e.what() << std::endl;
        return applyBilateralFilter(input, output); // Fallback to standard bilateral
    }
}

bool SelectiveBilateral::applyMultiscaleBilateral(const cv::Mat& input, cv::Mat& output) {
    try {
        // Initialize output with input
        input.copyTo(output);
        
        // Create multi-scale bilateral pyramid
        std::vector<cv::Mat> scales;
        scales.push_back(input.clone());
        
        // Downsample to create pyramid
        for (int i = 1; i < m_config.num_scales; i++) {
            cv::Mat downsampled;
            cv::pyrDown(scales[i-1], downsampled);
            scales.push_back(downsampled);
        }
        
        // Process each scale with adjusted parameters
        std::vector<cv::Mat> processed_scales;
        for (int i = 0; i < scales.size(); i++) {
            // Adjust parameters for each scale
            int diameter = m_config.diameter;
            double sigma_color = m_config.sigma_color * (1.0 + 0.5 * i); // Increase for coarser scales
            double sigma_space = m_config.sigma_space * (1.0 + 0.5 * i); // Increase for coarser scales
            
            // Apply bilateral filter to this scale
            cv::Mat filtered;
            if (m_config.use_gpu) {
                #ifdef WITH_CUDA
                    cv::cuda::GpuMat d_input, d_output;
                    d_input.upload(input);
                    // CUDA bilateral via free function:
                    cv::cuda::bilateralFilter(d_input, d_output,
                                            diameter,
                                            sigma_color,
                                            sigma_space);
                    d_output.download(output);
                #else
                    cv::bilateralFilter(input, output, diameter, sigma_color, sigma_space);
                #endif
            
            } else {
                // CPU implementation
                cv::bilateralFilter(scales[i], filtered, diameter, sigma_color, sigma_space);
            }
            
            processed_scales.push_back(filtered);
        }
        
        // Upsample and blend from coarse to fine
        for (int i = processed_scales.size() - 1; i > 0; i--) {
            cv::Mat upsampled;
            cv::pyrUp(processed_scales[i], upsampled, processed_scales[i-1].size());
            
            // Create detail mask for selective blending
            cv::Mat detail_mask;
            createDetailMask(processed_scales[i-1], detail_mask);
            
            // Blend based on detail mask
            for (int y = 0; y < processed_scales[i-1].rows; y++) {
                for (int x = 0; x < processed_scales[i-1].cols; x++) {
                    float detail_value = detail_mask.at<float>(y, x);
                    float coarse_weight = 1.0f - detail_value; // Less influence on detailed areas
                    
                    if (processed_scales[i-1].channels() == 1) {
                        // Grayscale image
                        float fine_value = processed_scales[i-1].at<uchar>(y, x);
                        float coarse_value = upsampled.at<uchar>(y, x);
                        
                        // Blend using detail-weighted average
                        processed_scales[i-1].at<uchar>(y, x) = cv::saturate_cast<uchar>(
                            fine_value * detail_value + coarse_value * coarse_weight
                        );
                    } else {
                        // Color image (assumed BGR)
                        cv::Vec3b fine_value = processed_scales[i-1].at<cv::Vec3b>(y, x);
                        cv::Vec3b coarse_value = upsampled.at<cv::Vec3b>(y, x);
                        
                        // Blend each channel
                        cv::Vec3b result;
                        for (int c = 0; c < 3; c++) {
                            result[c] = cv::saturate_cast<uchar>(
                                fine_value[c] * detail_value + coarse_value[c] * coarse_weight
                            );
                        }
                        
                        processed_scales[i-1].at<cv::Vec3b>(y, x) = result;
                    }
                }
            }
        }
        
        // The finest level is the output
        processed_scales[0].copyTo(output);
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error applying multi-scale bilateral filter: " << e.what() << std::endl;
        return applyBilateralFilter(input, output); // Fallback to standard bilateral
    }
}

bool SelectiveBilateral::createDetailMask(const cv::Mat& input, cv::Mat& detail_mask) {
    try {
        // Convert to grayscale if needed
        cv::Mat gray;
        if (input.channels() == 1) {
            input.copyTo(gray);
        } else {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        }
        
        // Calculate gradient magnitude using Sobel operators
        cv::Mat grad_x, grad_y;
        cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
        cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);
        
        // Calculate gradient magnitude
        cv::Mat magnitude;
        cv::magnitude(grad_x, grad_y, magnitude);
        
        // Calculate texture using local standard deviation
        cv::Mat mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        
        cv::Mat local_mean, local_stddev;
        int kernel_size = 5; // Size of local neighborhood
        cv::blur(gray, local_mean, cv::Size(kernel_size, kernel_size));
        
        // Calculate local variance
        cv::Mat diff_sq;
        cv::subtract(gray, local_mean, diff_sq);
        cv::multiply(diff_sq, diff_sq, diff_sq);
        
        cv::Mat local_var;
        cv::blur(diff_sq, local_var, cv::Size(kernel_size, kernel_size));
        
        // Local standard deviation
        cv::Mat texture;
        cv::sqrt(local_var, texture);
        
        // Combine gradient magnitude and texture for detail mask
        // Normalize both to 0-1 range
        double min_val, max_val;
        cv::minMaxLoc(magnitude, &min_val, &max_val);
        cv::Mat norm_magnitude = magnitude / max_val;
        
        cv::minMaxLoc(texture, &min_val, &max_val);
        cv::Mat norm_texture = texture / max_val;
        
        // Weighted combination
        detail_mask = 0.7 * norm_magnitude + 0.3 * norm_texture;
        
        // Threshold and create mask with smooth transition
        double threshold = m_config.detail_threshold / 255.0;
        cv::Mat mask = cv::Mat(detail_mask.size(), CV_32F);
        
        for (int y = 0; y < mask.rows; y++) {
            for (int x = 0; x < mask.cols; x++) {
                float detail_value = detail_mask.at<float>(y, x);
                
                // Apply sigmoid-like function centered at threshold
                float mask_value = 1.0f / (1.0f + std::exp(-(detail_value - threshold) * 10.0f));
                mask.at<float>(y, x) = mask_value;
            }
        }
        
        // Apply Gaussian blur for smoother transitions
        cv::GaussianBlur(mask, detail_mask, cv::Size(5, 5), 1.0);
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error creating detail mask: " << e.what() << std::endl;
        detail_mask = cv::Mat(input.size(), CV_32F, cv::Scalar(0.5)); // Neutral mask
        return false;
    }
}

bool SelectiveBilateral::calculateAdaptiveParams(const cv::Mat& input, 
                                               int& diameter, 
                                               double& sigma_color, 
                                               double& sigma_space) {
    try {
        // Calculate image statistics
        cv::Scalar mean, stddev;
        cv::meanStdDev(input, mean, stddev);
        
        // Average standard deviation across channels
        double avg_stddev = 0;
        for (int i = 0; i < input.channels(); i++) {
            avg_stddev += stddev[i];
        }
        avg_stddev /= input.channels();
        
        // Calculate noise level (approximation)
        double noise_level = avg_stddev;
        
        // Adjust parameters based on noise level and image content
        double noise_factor = 1.0;
        if (noise_level < 5.0) {
            noise_factor = 0.7; // Low noise
        } else if (noise_level > 15.0) {
            noise_factor = 1.5; // High noise
        }
        
        // Adjust parameters
        diameter = std::max(5, static_cast<int>(m_config.diameter * noise_factor));
        
        // Make sure diameter is odd
        if (diameter % 2 == 0) {
            diameter++;
        }
        
        // Constrain maximum diameter for performance
        diameter = std::min(diameter, 15);
        
        // Adjust sigma values
        sigma_color = m_config.sigma_color * noise_factor;
        sigma_space = m_config.sigma_space * noise_factor;
        
        // Different parameters for different stages
        if (m_config.stage == POST_PROCESSING) {
            // For post-processing, be more conservative to avoid oversmoothing
            diameter = std::max(3, diameter - 2);
            sigma_color *= 0.8;
            sigma_space *= 0.8;
        }
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error calculating adaptive parameters: " << e.what() << std::endl;
        // Keep original parameters
        return false;
    }
}

bool SelectiveBilateral::applyJointBilateral(const cv::Mat& input, 
                                            const cv::Mat& detail_mask, 
                                            cv::Mat& output) {
    cv::Mat filtered;
    try {
        // Standard bilateral filtering
        if (!applyBilateralFilter(input, filtered)) {
            input.copyTo(output);
            return false;
        }
        
        // Initialize output
        output = cv::Mat(input.size(), input.type());
        
        // Blend original and filtered based on detail mask
        if (input.channels() == 1) {
            // Grayscale image
            for (int y = 0; y < output.rows; y++) {
                for (int x = 0; x < output.cols; x++) {
                    float detail_value = detail_mask.at<float>(y, x);
                    float preservation = 1.0f + detail_value * (m_config.edge_preserve - 1.0f);
                    
                    float original = input.at<uchar>(y, x);
                    float filtered_value = filtered.at<uchar>(y, x);
                    
                    // Blend with detail-dependent weights
                    output.at<uchar>(y, x) = cv::saturate_cast<uchar>(
                        original * detail_value * preservation + 
                        filtered_value * (1.0f - detail_value * preservation)
                    );
                }
            }
        } else {
            // Color image (assumed BGR)
            for (int y = 0; y < output.rows; y++) {
                for (int x = 0; x < output.cols; x++) {
                    float detail_value = detail_mask.at<float>(y, x);
                    float preservation = 1.0f + detail_value * (m_config.edge_preserve - 1.0f);
                    
                    cv::Vec3b original = input.at<cv::Vec3b>(y, x);
                    cv::Vec3b filtered_value = filtered.at<cv::Vec3b>(y, x);
                    
                    // Blend each channel with detail-dependent weights
                    cv::Vec3b result;
                    for (int c = 0; c < 3; c++) {
                        result[c] = cv::saturate_cast<uchar>(
                            original[c] * detail_value * preservation + 
                            filtered_value[c] * (1.0f - detail_value * preservation)
                        );
                    }
                    
                    output.at<cv::Vec3b>(y, x) = result;
                }
            }
        }
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error applying joint bilateral filter: " << e.what() << std::endl;
        if (!filtered.empty()) {
            filtered.copyTo(output); // Use standard bilateral if joint fails
        } else {
            input.copyTo(output);
        }
        return false;
    }
}