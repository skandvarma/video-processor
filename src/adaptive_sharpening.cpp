#include "adaptive_sharpening.h"
#include <iostream>
#include <algorithm>

// Check if OpenCV was built with CUDA support
#ifdef WITH_CUDA
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

AdaptiveSharpening::AdaptiveSharpening() 
    : m_initialized(false) {
}

AdaptiveSharpening::AdaptiveSharpening(const Config& config)
    : m_config(config), m_initialized(false) {
}

AdaptiveSharpening::~AdaptiveSharpening() {
}

bool AdaptiveSharpening::initialize() {
    // Check if GPU is available when requested
    if (m_config.use_gpu) {
#ifdef WITH_CUDA
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
            std::cout << "CUDA requested for adaptive sharpening but not available. Using CPU fallback." << std::endl;
            m_config.use_gpu = false;
        } else {
            std::cout << "Using CUDA for adaptive sharpening." << std::endl;
        }
#else
        std::cout << "CUDA requested for adaptive sharpening but OpenCV was built without CUDA support. Using CPU fallback." << std::endl;
        m_config.use_gpu = false;
#endif
    }
    
    m_initialized = true;
    return true;
}

void AdaptiveSharpening::setConfig(const Config& config) {
    m_config = config;
}

AdaptiveSharpening::Config AdaptiveSharpening::getConfig() const {
    return m_config;
}

bool AdaptiveSharpening::process(const cv::Mat& input, cv::Mat& output) {
    if (!m_initialized) {
        std::cerr << "Adaptive sharpening module not initialized" << std::endl;
        return false;
    }
    
    if (input.empty()) {
        std::cerr << "Empty input image" << std::endl;
        return false;
    }
    
    try {
        // Create edge mask for adaptive sharpening
        cv::Mat edge_mask;
        if (!createEdgeMask(input, edge_mask)) {
            std::cerr << "Failed to create edge mask" << std::endl;
            return false;
        }
        
        // Apply unsharp mask using the edge mask
        if (m_config.adaptive_sigma) {
            // Calculate texture map
            cv::Mat texture_map;
            if (!calculateTextureMap(input, texture_map)) {
                std::cerr << "Failed to calculate texture map" << std::endl;
                return false;
            }
            
            // Calculate adaptive sigma values
            cv::Mat sigma_map;
            if (!calculateAdaptiveSigma(texture_map, sigma_map)) {
                std::cerr << "Failed to calculate adaptive sigma values" << std::endl;
                return false;
            }
            
            // Apply unsharp mask with variable sigma
            if (!applyVariableSigmaUnsharpMask(input, sigma_map, edge_mask, output)) {
                std::cerr << "Failed to apply variable sigma unsharp mask" << std::endl;
                return false;
            }
        } else {
            // Apply standard unsharp mask
            if (!applyUnsharpMask(input, edge_mask, output)) {
                std::cerr << "Failed to apply unsharp mask" << std::endl;
                return false;
            }
        }
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error in adaptive sharpening: " << e.what() << std::endl;
        input.copyTo(output);
        return false;
    }
}

bool AdaptiveSharpening::createEdgeMask(const cv::Mat& input, cv::Mat& edge_mask) {
    try {
        // Convert to grayscale if needed
        cv::Mat gray;
        if (input.channels() == 1) {
            input.copyTo(gray);
        } else {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        }
        
        // Detect edges using multiple approaches for better results
        
        // 1. Sobel edge detection
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        cv::Mat sobel_grad;
        
        // Use GPU if available and requested
        if (m_config.use_gpu) {
#ifdef WITH_CUDA
            cv::cuda::GpuMat d_gray, d_grad_x, d_grad_y, d_abs_grad_x, d_abs_grad_y, d_sobel_grad;
            d_gray.upload(gray);
            
            // Sobel gradients
            cv::Ptr<cv::cuda::Filter> sobel_x = cv::cuda::createSobelFilter(CV_8UC1, CV_16S, 1, 0, 3);
            cv::Ptr<cv::cuda::Filter> sobel_y = cv::cuda::createSobelFilter(CV_8UC1, CV_16S, 0, 1, 3);
            
            sobel_x->apply(d_gray, d_grad_x);
            sobel_y->apply(d_gray, d_grad_y);
            
            // Convert to absolute values
            cv::cuda::abs(d_grad_x, d_abs_grad_x);
            cv::cuda::abs(d_grad_y, d_abs_grad_y);
            
            // Combine gradients
            cv::cuda::addWeighted(d_abs_grad_x, 0.5, d_abs_grad_y, 0.5, 0, d_sobel_grad);
            d_sobel_grad.download(sobel_grad);
            
            // Convert to 8-bit
            sobel_grad.convertTo(sobel_grad, CV_8UC1);
#else
            // CPU fallback
            cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
            cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
            
            cv::convertScaleAbs(grad_x, abs_grad_x);
            cv::convertScaleAbs(grad_y, abs_grad_y);
            
            cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_grad);
#endif
        } else {
            // CPU implementation
            cv::Sobel(gray, grad_x, CV_16S, 1, 0, 3);
            cv::Sobel(gray, grad_y, CV_16S, 0, 1, 3);
            
            cv::convertScaleAbs(grad_x, abs_grad_x);
            cv::convertScaleAbs(grad_y, abs_grad_y);
            
            cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel_grad);
        }
        
        // 2. Laplacian edge detection
        cv::Mat laplacian;
        if (m_config.use_gpu) {
#ifdef WITH_CUDA
            cv::cuda::GpuMat d_gray, d_laplacian;
            d_gray.upload(gray);
            
            // Laplacian filter
            cv::Ptr<cv::cuda::Filter> laplacian_filter = cv::cuda::createLaplacianFilter(CV_8UC1, CV_16S, 3);
            laplacian_filter->apply(d_gray, d_laplacian);
            
            // Convert to absolute values
            cv::cuda::abs(d_laplacian, d_laplacian);
            d_laplacian.download(laplacian);
            
            // Convert to 8-bit
            laplacian.convertTo(laplacian, CV_8UC1);
#else
            // CPU fallback
            cv::Laplacian(gray, laplacian, CV_16S, 3);
            cv::convertScaleAbs(laplacian, laplacian);
#endif
        } else {
            // CPU implementation
            cv::Laplacian(gray, laplacian, CV_16S, 3);
            cv::convertScaleAbs(laplacian, laplacian);
        }
        
        // 3. Combine edge detectors for better results
        cv::Mat combined_edges;
        cv::addWeighted(sobel_grad, 0.6, laplacian, 0.4, 0, combined_edges);
        
        // 4. Create edge mask with smooth transition using threshold
        cv::Mat edge_mask_8u;
        double threshold = m_config.edge_threshold;
        
        // Threshold with smooth falloff for natural transitions
        cv::Mat edge_mask_float;
        combined_edges.convertTo(edge_mask_float, CV_32F, 1.0/255.0);
        
        // Create sigmoid-like mask with threshold as midpoint
        edge_mask = cv::Mat(edge_mask_float.size(), CV_32F);
        float scale_factor = 0.1f; // Controls transition steepness
        
        for (int y = 0; y < edge_mask.rows; y++) {
            for (int x = 0; x < edge_mask.cols; x++) {
                float edge_value = edge_mask_float.at<float>(y, x) * 255.0f;
                // Sigmoid-like function centered at threshold
                float sigmoid = 1.0f / (1.0f + std::exp(-(edge_value - threshold) * scale_factor));
                edge_mask.at<float>(y, x) = sigmoid;
            }
        }
        
        // Apply slight blur to the mask for smoother transitions
        cv::GaussianBlur(edge_mask, edge_mask, cv::Size(5, 5), 1.5);
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error creating edge mask: " << e.what() << std::endl;
        edge_mask = cv::Mat(input.size(), CV_32F, cv::Scalar(0.5)); // Neutral mask
        return false;
    }
}

bool AdaptiveSharpening::applyUnsharpMask(const cv::Mat& input, const cv::Mat& edge_mask, cv::Mat& output) {
    try {
        // Create a blurred version of the input
        cv::Mat blurred;
        
        if (m_config.use_gpu) {
#ifdef WITH_CUDA
            cv::cuda::GpuMat d_input, d_blurred;
            d_input.upload(input);
            
            // Apply Gaussian blur
            cv::Ptr<cv::cuda::Filter> blur_filter = cv::cuda::createGaussianFilter(
                input.type(), input.type(), 
                cv::Size(m_config.kernel_size, m_config.kernel_size), 
                m_config.sigma);
                
            blur_filter->apply(d_input, d_blurred);
            d_blurred.download(blurred);
#else
            // CPU fallback
            cv::GaussianBlur(input, blurred, 
                cv::Size(m_config.kernel_size, m_config.kernel_size), 
                m_config.sigma);
#endif
        } else {
            // CPU implementation
            cv::GaussianBlur(input, blurred, 
                cv::Size(m_config.kernel_size, m_config.kernel_size), 
                m_config.sigma);
        }
        
        // Create the unsharp mask by subtracting blurred from original
        cv::Mat unsharp_mask;
        cv::subtract(input, blurred, unsharp_mask);
        
        // Apply variable strength based on edge mask
        cv::Mat weighted_mask;
        
        // Create an output image of the same type as input
        output = cv::Mat(input.size(), input.type());
        
        // Process based on number of channels
        if (input.channels() == 1) {
            // For grayscale images
            for (int y = 0; y < output.rows; y++) {
                for (int x = 0; x < output.cols; x++) {
                    // Calculate adaptive strength based on edge mask
                    float edge_value = edge_mask.at<float>(y, x);
                    float strength = m_config.strength * (
                        edge_value * m_config.edge_strength + 
                        (1.0f - edge_value) * m_config.smooth_strength
                    );
                    
                    // Apply unsharp mask with adaptive strength
                    float original = input.at<uchar>(y, x);
                    float mask_value = unsharp_mask.at<uchar>(y, x);
                    
                    // Apply sharpening
                    float sharpened = original + strength * mask_value;
                    
                    // Clamp to valid range
                    output.at<uchar>(y, x) = cv::saturate_cast<uchar>(sharpened);
                }
            }
        } else {
            // For color images (assumed BGR)
            for (int y = 0; y < output.rows; y++) {
                for (int x = 0; x < output.cols; x++) {
                    // Calculate adaptive strength based on edge mask
                    float edge_value = edge_mask.at<float>(y, x);
                    float strength = m_config.strength * (
                        edge_value * m_config.edge_strength + 
                        (1.0f - edge_value) * m_config.smooth_strength
                    );
                    
                    // Apply unsharp mask with adaptive strength to each channel
                    cv::Vec3b original = input.at<cv::Vec3b>(y, x);
                    cv::Vec3b mask_value = unsharp_mask.at<cv::Vec3b>(y, x);
                    
                    // Apply sharpening to each channel
                    cv::Vec3b sharpened;
                    for (int c = 0; c < 3; c++) {
                        sharpened[c] = cv::saturate_cast<uchar>(
                            original[c] + strength * mask_value[c]
                        );
                    }
                    
                    // Set output pixel
                    output.at<cv::Vec3b>(y, x) = sharpened;
                }
            }
        }
        
        // Apply tone preservation if enabled
        if (m_config.preserve_tone) {
            // Convert to YCrCb color space
            cv::Mat ycrcb_input, ycrcb_output;
            cv::cvtColor(input, ycrcb_input, cv::COLOR_BGR2YCrCb);
            cv::cvtColor(output, ycrcb_output, cv::COLOR_BGR2YCrCb);
            
            // Split into channels
            std::vector<cv::Mat> channels_input, channels_output;
            cv::split(ycrcb_input, channels_input);
            cv::split(ycrcb_output, channels_output);
            
            // Only apply sharpening to Y channel, preserve Cr and Cb
            channels_output[0] = channels_output[0];  // Y (already sharpened)
            channels_output[1] = channels_input[1];   // Cr (preserved)
            channels_output[2] = channels_input[2];   // Cb (preserved)
            
            // Merge channels
            cv::merge(channels_output, ycrcb_output);
            
            // Convert back to BGR
            cv::cvtColor(ycrcb_output, output, cv::COLOR_YCrCb2BGR);
        }
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error applying unsharp mask: " << e.what() << std::endl;
        input.copyTo(output);
        return false;
    }
}

bool AdaptiveSharpening::calculateTextureMap(const cv::Mat& input, cv::Mat& texture_map) {
    try {
        // Convert to grayscale if needed
        cv::Mat gray;
        if (input.channels() == 1) {
            input.copyTo(gray);
        } else {
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        }
        
        // Use local standard deviation as a measure of texture
        cv::Mat mean, stddev;
        cv::Mat kernel = cv::Mat::ones(7, 7, CV_32F) / 49.0f;
        
        // Mean filter
        cv::Mat local_mean;
        cv::filter2D(gray, local_mean, -1, kernel);
        
        // Calculate squared differences
        cv::Mat diff;
        cv::subtract(gray, local_mean, diff);
        cv::Mat diff_sq;
        cv::multiply(diff, diff, diff_sq);
        
        // Mean of squared differences
        cv::Mat local_var;
        cv::filter2D(diff_sq, local_var, -1, kernel);
        
        // Standard deviation (sqrt of variance)
        cv::sqrt(local_var, texture_map);
        
        // Normalize to 0-1 range
        double min_val, max_val;
        cv::minMaxLoc(texture_map, &min_val, &max_val);
        texture_map = (texture_map - min_val) / (max_val - min_val);
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error calculating texture map: " << e.what() << std::endl;
        texture_map = cv::Mat(input.size(), CV_32F, cv::Scalar(0.5)); // Neutral map
        return false;
    }
}

bool AdaptiveSharpening::calculateAdaptiveSigma(const cv::Mat& texture_map, cv::Mat& sigma_map) {
    try {
        // Create sigma map of same size as texture map
        sigma_map = cv::Mat(texture_map.size(), CV_32F);
        
        // Define sigma range
        float min_sigma = 0.8f;
        float max_sigma = 2.5f;
        
        // Map texture values to sigma values
        // High texture areas get smaller sigma (more precise sharpening)
        // Low texture areas get larger sigma (more spread-out sharpening)
        for (int y = 0; y < sigma_map.rows; y++) {
            for (int x = 0; x < sigma_map.cols; x++) {
                float texture_value = texture_map.at<float>(y, x);
                
                // Inverse mapping: high texture -> low sigma, low texture -> high sigma
                float sigma = max_sigma - texture_value * (max_sigma - min_sigma);
                
                // Store in sigma map
                sigma_map.at<float>(y, x) = sigma;
            }
        }
        
        // Apply slight blur to sigma map for smoother transitions
        cv::GaussianBlur(sigma_map, sigma_map, cv::Size(5, 5), 1.0);
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error calculating adaptive sigma: " << e.what() << std::endl;
        sigma_map = cv::Mat(texture_map.size(), CV_32F, cv::Scalar(m_config.sigma)); // Default sigma
        return false;
    }
}

bool AdaptiveSharpening::applyVariableSigmaUnsharpMask(const cv::Mat& input, 
                                                     const cv::Mat& sigma_map, 
                                                     const cv::Mat& edge_mask, 
                                                     cv::Mat& output) {
    try {
        // Create a blurred version of the input with variable sigma
        cv::Mat blurred = cv::Mat::zeros(input.size(), input.type());
        
        // Apply blur in a more efficient way using integral images
        // First create a set of blurred images with different sigmas
        const int num_sigma_levels = 5;
        std::vector<cv::Mat> blurred_levels(num_sigma_levels);
        
        double min_sigma_d, max_sigma_d;
        cv::minMaxLoc(sigma_map, &min_sigma_d, &max_sigma_d);
        float min_sigma = static_cast<float>(min_sigma_d);
        float max_sigma = static_cast<float>(max_sigma_d);

        
        // Create blurred versions with different sigma values
        for (int i = 0; i < num_sigma_levels; i++) {
            float sigma = min_sigma + (max_sigma - min_sigma) * i / (num_sigma_levels - 1);
            cv::GaussianBlur(input, blurred_levels[i], 
                cv::Size(m_config.kernel_size, m_config.kernel_size), 
                sigma);
        }
        
        // Interpolate between blurred images based on sigma map
        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.cols; x++) {
                float sigma = sigma_map.at<float>(y, x);
                
                // Find the two closest sigma levels
                int idx_low = static_cast<int>((sigma - min_sigma) / (max_sigma - min_sigma) * (num_sigma_levels - 1));
                idx_low = std::max(0, std::min(idx_low, num_sigma_levels - 2));
                int idx_high = idx_low + 1;
                
                // Get sigmas for the two levels
                float sigma_low = min_sigma + (max_sigma - min_sigma) * idx_low / (num_sigma_levels - 1);
                float sigma_high = min_sigma + (max_sigma - min_sigma) * idx_high / (num_sigma_levels - 1);
                
                // Calculate interpolation factor
                float alpha = (sigma - sigma_low) / (sigma_high - sigma_low);
                
                // Interpolate blurred pixels
                if (input.channels() == 1) {
                    uchar pixel_low = blurred_levels[idx_low].at<uchar>(y, x);
                    uchar pixel_high = blurred_levels[idx_high].at<uchar>(y, x);
                    blurred.at<uchar>(y, x) = static_cast<uchar>(
                        pixel_low * (1.0f - alpha) + pixel_high * alpha
                    );
                } else {
                    cv::Vec3b pixel_low = blurred_levels[idx_low].at<cv::Vec3b>(y, x);
                    cv::Vec3b pixel_high = blurred_levels[idx_high].at<cv::Vec3b>(y, x);
                    
                    cv::Vec3b result;
                    for (int c = 0; c < 3; c++) {
                        result[c] = static_cast<uchar>(
                            pixel_low[c] * (1.0f - alpha) + pixel_high[c] * alpha
                        );
                    }
                    
                    blurred.at<cv::Vec3b>(y, x) = result;
                }
            }
        }
        
        // Create the unsharp mask by subtracting blurred from original
        cv::Mat unsharp_mask;
        cv::subtract(input, blurred, unsharp_mask);
        
        // Create output image
        output = cv::Mat(input.size(), input.type());
        
        // Apply variable strength based on edge mask
        if (input.channels() == 1) {
            // For grayscale images
            for (int y = 0; y < output.rows; y++) {
                for (int x = 0; x < output.cols; x++) {
                    // Calculate adaptive strength based on edge mask
                    float edge_value = edge_mask.at<float>(y, x);
                    float strength = m_config.strength * (
                        edge_value * m_config.edge_strength + 
                        (1.0f - edge_value) * m_config.smooth_strength
                    );
                    
                    // Apply unsharp mask with adaptive strength
                    float original = input.at<uchar>(y, x);
                    float mask_value = unsharp_mask.at<uchar>(y, x);
                    
                    // Apply sharpening
                    float sharpened = original + strength * mask_value;
                    
                    // Clamp to valid range
                    output.at<uchar>(y, x) = cv::saturate_cast<uchar>(sharpened);
                }
            }
        } else {
            // For color images (assumed BGR)
            for (int y = 0; y < output.rows; y++) {
                for (int x = 0; x < output.cols; x++) {
                    // Calculate adaptive strength based on edge mask
                    float edge_value = edge_mask.at<float>(y, x);
                    float strength = m_config.strength * (
                        edge_value * m_config.edge_strength + 
                        (1.0f - edge_value) * m_config.smooth_strength
                    );
                    
                    // Apply unsharp mask with adaptive strength to each channel
                    cv::Vec3b original = input.at<cv::Vec3b>(y, x);
                    cv::Vec3b mask_value = unsharp_mask.at<cv::Vec3b>(y, x);
                    
                    // Apply sharpening to each channel
                    cv::Vec3b sharpened;
                    for (int c = 0; c < 3; c++) {
                        sharpened[c] = cv::saturate_cast<uchar>(
                            original[c] + strength * mask_value[c]
                        );
                    }
                    
                    // Set output pixel
                    output.at<cv::Vec3b>(y, x) = sharpened;
                }
            }
        }
        
        // Apply tone preservation if enabled
        if (m_config.preserve_tone && input.channels() == 3) {
            // Convert to YCrCb color space
            cv::Mat ycrcb_input, ycrcb_output;
            cv::cvtColor(input, ycrcb_input, cv::COLOR_BGR2YCrCb);
            cv::cvtColor(output, ycrcb_output, cv::COLOR_BGR2YCrCb);
            
            // Split into channels
            std::vector<cv::Mat> channels_input, channels_output;
            cv::split(ycrcb_input, channels_input);
            cv::split(ycrcb_output, channels_output);
            
            // Only apply sharpening to Y channel, preserve Cr and Cb
            channels_output[0] = channels_output[0];  // Y (already sharpened)
            channels_output[1] = channels_input[1];   // Cr (preserved)
            channels_output[2] = channels_input[2];   // Cb (preserved)
            
            // Merge channels
            cv::merge(channels_output, ycrcb_output);
            
            // Convert back to BGR
            cv::cvtColor(ycrcb_output, output, cv::COLOR_YCrCb2BGR);
        }
        
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error applying variable sigma unsharp mask: " << e.what() << std::endl;
        input.copyTo(output);
        return false;
    }
}