#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

/**
 * @brief Implements selective bilateral filtering for video upscaling
 * 
 * This class provides edge-preserving smoothing with configurable
 * parameters that can be used as both pre-processing and post-processing
 * stages in the video upscaling pipeline.
 */
class SelectiveBilateral {
public:
    enum FilteringStage {
        PRE_PROCESSING,   // Applied before upscaling
        POST_PROCESSING   // Applied after upscaling
    };
    
    struct Config {
        // General parameters
        FilteringStage stage = PRE_PROCESSING;
        bool use_gpu = true;                // Use GPU acceleration if available
        bool adaptive_params = true;        // Use content-adaptive parameters
        
        // Bilateral filter parameters
        int diameter = 7;                   // Diameter of pixel neighborhood
        double sigma_color = 30.0;          // Filter sigma in color space
        double sigma_space = 30.0;          // Filter sigma in coordinate space
        
        // Selective processing parameters
        bool selective = true;              // Apply filter selectively based on content
        double detail_threshold = 30.0;     // Threshold for preserving detailed areas
        double texture_boost = 1.5;         // Boost factor for textured areas
        double edge_preserve = 2.0;         // Factor for edge preservation
        
        // Multi-scale parameters
        bool use_multiscale = true;         // Use multi-scale bilateral filtering
        int num_scales = 3;                 // Number of scales for multi-scale filtering
    };
    
    /**
     * @brief Construct a new Selective Bilateral object with default configuration
     */
    SelectiveBilateral();
    
    /**
     * @brief Construct a new Selective Bilateral object with custom configuration
     * 
     * @param config Configuration parameters
     */
    explicit SelectiveBilateral(const Config& config);
    
    /**
     * @brief Destroy the Selective Bilateral object
     */
    ~SelectiveBilateral();
    
    /**
     * @brief Initialize the selective bilateral filtering module
     * 
     * @return true if initialization was successful
     */
    bool initialize();
    
    /**
     * @brief Apply selective bilateral filtering to an image
     * 
     * @param input The input image
     * @param output The output filtered image
     * @return true if processing was successful
     */
    bool process(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief Set the configuration parameters
     * 
     * @param config The new configuration
     */
    void setConfig(const Config& config);
    
    /**
     * @brief Get the current configuration
     * 
     * @return The current configuration
     */
    Config getConfig() const;
    
private:
    Config m_config;
    bool m_initialized;
    
    /**
     * @brief Apply standard bilateral filter
     * 
     * @param input The input image
     * @param output The output filtered image
     * @return true if successful
     */
    bool applyBilateralFilter(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief Apply selective bilateral filter based on content
     * 
     * @param input The input image
     * @param output The output filtered image
     * @return true if successful
     */
    bool applySelectiveBilateral(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief Apply multi-scale bilateral filter
     * 
     * @param input The input image
     * @param output The output filtered image
     * @return true if successful
     */
    bool applyMultiscaleBilateral(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief Create a detail mask for selective filtering
     * 
     * @param input The input image
     * @param detail_mask Output detail mask (0-1 float values)
     * @return true if successful
     */
    bool createDetailMask(const cv::Mat& input, cv::Mat& detail_mask);
    
    /**
     * @brief Calculate adaptive filtering parameters based on content
     * 
     * @param input The input image
     * @param diameter Output diameter parameter
     * @param sigma_color Output sigma color parameter
     * @param sigma_space Output sigma space parameter
     * @return true if successful
     */
    bool calculateAdaptiveParams(const cv::Mat& input, 
                                int& diameter, 
                                double& sigma_color, 
                                double& sigma_space);
    
    /**
     * @brief Apply joint bilateral filter with a detail mask
     * 
     * @param input The input image
     * @param detail_mask The detail mask (0-1 float values)
     * @param output The output filtered image
     * @return true if successful
     */
    bool applyJointBilateral(const cv::Mat& input, 
                             const cv::Mat& detail_mask, 
                             cv::Mat& output);
};