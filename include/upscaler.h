#pragma once

#include <opencv2/opencv.hpp>
#include "dnn_super_res.h"
#include <memory>
#include <string>

// Forward declaration of implementation class
class UpscalerImpl;

/**
 * @brief Class for video frame upscaling with GPU acceleration when available
 * 
 * This class handles upscaling of video frames using various algorithms,
 * with optimized GPU implementation when CUDA is available.
 */
class Upscaler {
public:
    /**
     * @brief Upscaling algorithm to use
     */
    enum Algorithm {
        NEAREST,      ///< Nearest neighbor (fastest, lowest quality)
        BILINEAR,     ///< Bilinear interpolation (good balance)
        BICUBIC,      ///< Bicubic interpolation (better quality)
        LANCZOS,      ///< Lanczos interpolation (highest quality, slowest)
        SUPER_RES     ///< Super resolution with image enhancement (highest quality)
    };
    
    /**
     * @brief Construct a new Upscaler
     * @param algorithm The upscaling algorithm to use
     * @param use_gpu Whether to use GPU acceleration if available
     */
    explicit Upscaler(Algorithm algorithm = BILINEAR, bool use_gpu = true);
    
    /**
     * @brief Destroy the Upscaler object and free resources
     */
    ~Upscaler();
    
    /**
     * @brief Initialize the upscaler with target resolution
     * @param target_width Target width for upscaled frames
     * @param target_height Target height for upscaled frames
     * @return true if initialization was successful
     */
    bool initialize(int target_width, int target_height);
    
    /**
     * @brief Upscale an input frame to the target resolution
     * @param input Input frame to upscale
     * @param output Output frame (will be resized to target resolution)
     * @return true if upscaling was successful
     */
    bool upscale(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief Set the upscaling algorithm
     * @param algorithm The algorithm to use
     */
    void setAlgorithm(Algorithm algorithm);
    
    /**
     * @brief Enable or disable GPU acceleration
     * @param use_gpu true to enable GPU, false to use CPU
     * @return true if the requested mode was set
     */
    bool setUseGPU(bool use_gpu);
    
    /**
     * @brief Check if GPU acceleration is being used
     * @return true if using GPU, false if using CPU
     */
    bool isUsingGPU() const;
    
    /**
     * @brief Get a string description of the current algorithm
     * @return String name of the algorithm
     */
    std::string getAlgorithmName() const;
    
    /**
     * @brief Check if GPU acceleration is available on this system
     * @return true if GPU acceleration is available
     */
    static bool isGPUAvailable();

    bool adjustQualityForPerformance(double processing_time, double target_time);

    
private:
    Algorithm m_algorithm;             // Selected upscaling algorithm
    bool m_use_gpu;                    // Whether to use GPU acceleration
    bool m_initialized;                // Whether upscaler has been initialized
    int m_target_width;                // Target width for upscaled frames
    int m_target_height;               // Target height for upscaled frames
    
    // Implementation of upscaler (CPU or GPU depending on availability)
    std::unique_ptr<UpscalerImpl> m_impl;
    std::unique_ptr<DnnSuperRes> m_dnn_sr;
    
    // Initialize the implementation based on current settings
    bool initializeImpl();
};