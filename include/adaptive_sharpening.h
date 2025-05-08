#pragma once

#include <opencv2/opencv.hpp>
#include <memory>

/**
 * @brief Implements adaptive sharpening for video upscaling
 * 
 * This class provides intelligent edge-aware sharpening with
 * configurable parameters to enhance details while avoiding noise amplification.
 */
class AdaptiveSharpening {
public:
    struct Config {
        float strength = 0.8f;            // Overall sharpening strength (0.0-1.0)
        float edge_strength = 1.2f;       // Edge sharpening strength multiplier
        float smooth_strength = 0.4f;     // Smooth area sharpening strength multiplier
        float edge_threshold = 30.0f;     // Threshold for edge detection
        float sigma = 1.5f;               // Gaussian blur sigma for unsharp mask
        int kernel_size = 5;              // Kernel size for unsharp mask
        bool preserve_tone = true;        // Preserve tone during sharpening
        bool use_gpu = true;              // Use GPU acceleration if available
        bool adaptive_sigma = true;       // Adaptively adjust sigma based on local texture
    };
    
    /**
     * @brief Construct a new Adaptive Sharpening object with default configuration
     */
    AdaptiveSharpening();
    
    /**
     * @brief Construct a new Adaptive Sharpening object with custom configuration
     * 
     * @param config Configuration parameters
     */
    explicit AdaptiveSharpening(const Config& config);
    
    /**
     * @brief Destroy the Adaptive Sharpening object
     */
    ~AdaptiveSharpening();
    
    /**
     * @brief Initialize the adaptive sharpening module
     * 
     * @return true if initialization was successful
     */
    bool initialize();
    
    /**
     * @brief Apply adaptive sharpening to an image
     * 
     * @param input The input image
     * @param output The output image with adaptive sharpening applied
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
     * @brief Create an edge mask for adaptive sharpening
     * 
     * @param input The input image
     * @param edge_mask Output edge mask (0-1 float values)
     * @return true if successful
     */
    bool createEdgeMask(const cv::Mat& input, cv::Mat& edge_mask);
    
    /**
     * @brief Apply unsharp mask with adaptive parameters
     * 
     * @param input The input image
     * @param edge_mask The edge mask
     * @param output The output sharpened image
     * @return true if successful
     */
    bool applyUnsharpMask(const cv::Mat& input, const cv::Mat& edge_mask, cv::Mat& output);
    
    /**
     * @brief Calculate texture energy map for adaptive sigma
     * 
     * @param input The input image
     * @param texture_map Output texture energy map
     * @return true if successful
     */
    bool calculateTextureMap(const cv::Mat& input, cv::Mat& texture_map);
    
    /**
     * @brief Calculate adaptive sigma values based on texture
     * 
     * @param texture_map Texture energy map
     * @param sigma_map Output map of sigma values
     * @return true if successful
     */
    bool calculateAdaptiveSigma(const cv::Mat& texture_map, cv::Mat& sigma_map);
    
    /**
     * @brief Apply unsharp mask with variable sigma
     * 
     * @param input The input image
     * @param sigma_map Map of sigma values
     * @param edge_mask The edge mask
     * @param output The output sharpened image
     * @return true if successful
     */
    bool applyVariableSigmaUnsharpMask(const cv::Mat& input, 
                                      const cv::Mat& sigma_map, 
                                      const cv::Mat& edge_mask, 
                                      cv::Mat& output);
};