#pragma once

#include <opencv2/opencv.hpp>
#include "dnn_super_res.h"
#include <memory>
#include <string>

// Forward declarations for enhancement modules
class SelectiveBilateral;
class AdaptiveSharpening;
class TemporalConsistency;

// Forward declaration of implementation classes
class UpscalerImpl;
class CPUImpl;
#ifdef WITH_CUDA
class GPUImpl;
#endif

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
        SUPER_RES,    ///< Super resolution with image enhancement (highest quality)
        REAL_ESRGAN   ///< RealESRGAN model (best quality)
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

    /**
     * @brief Adaptively adjust quality based on performance
     * @param processing_time Current processing time
     * @param target_time Target processing time
     * @return true if quality was adjusted
     */
    bool adjustQualityForPerformance(double processing_time, double target_time);
    
    /**
     * @brief Get the target width for upscaling
     * @return Target width in pixels
     */
    int getTargetWidth() const { return m_target_width; }

    /**
     * @brief Get the target height for upscaling
     * @return Target height in pixels
     */
    int getTargetHeight() const { return m_target_height; }

    // Enhancement control methods
    void setUseSelectiveBilateral(bool enable) { m_use_selective_bilateral = enable; }
    void setUseAdaptiveSharpening(bool enable) { m_use_adaptive_sharpening = enable; }
    void setUseTemporalConsistency(bool enable) { m_use_temporal_consistency = enable; }
    
    bool isUsingSelectiveBilateral() const { return m_use_selective_bilateral; }
    bool isUsingAdaptiveSharpening() const { return m_use_adaptive_sharpening; }
    bool isUsingTemporalConsistency() const { return m_use_temporal_consistency; }
    
    // Methods to access enhancement modules for parameter tuning
    SelectiveBilateral* getBilateralPreProcessor() { return m_bilateral_pre.get(); }
    AdaptiveSharpening* getAdaptiveSharpening() { return m_sharpening.get(); }
    SelectiveBilateral* getBilateralPostProcessor() { return m_bilateral_post.get(); }
    TemporalConsistency* getTemporalConsistency() { return m_temporal_consistency.get(); }
    
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
    
    // Helper method to initialize enhancements
    bool initializeEnhancements();
    
    // Enhancement modules
    std::unique_ptr<SelectiveBilateral> m_bilateral_pre;
    std::unique_ptr<AdaptiveSharpening> m_sharpening;
    std::unique_ptr<SelectiveBilateral> m_bilateral_post;
    std::unique_ptr<TemporalConsistency> m_temporal_consistency;
    
    // Enhancement flags
    bool m_use_selective_bilateral;
    bool m_use_adaptive_sharpening;
    bool m_use_temporal_consistency;
};