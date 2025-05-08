#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <mutex>
#include <memory>

/**
 * @brief Implements temporal consistency for video upscaling
 * 
 * This class manages a buffer of previous frames and implements
 * optical flow based alignment to provide temporally consistent
 * video upscaling with reduced flickering.
 */
class TemporalConsistency {
public:
    struct Config {
        int buffer_size = 3;                // Number of previous frames to buffer
        float blend_strength = 0.6f;        // Strength of temporal blending (0.0-1.0)
        float motion_threshold = 15.0f;     // Threshold for motion detection
        float scene_change_threshold = 100.0f; // Threshold for scene change detection
        bool use_gpu = true;                // Use GPU acceleration if available
        
        // Optical flow parameters
        double pyr_scale = 0.5;             // Pyramid scale
        int levels = 3;                     // Pyramid levels
        int winsize = 15;                   // Window size
        int iterations = 3;                 // Iterations
        int poly_n = 5;                     // Poly N
        double poly_sigma = 1.2;            // Poly Sigma
        int flags = 0;                      // Flags (can include OPTFLOW_FARNEBACK_GAUSSIAN)
    };
    
    /**
     * @brief Construct a new Temporal Consistency object with default configuration
     */
    TemporalConsistency();
    
    /**
     * @brief Construct a new Temporal Consistency object with custom configuration
     * 
     * @param config Configuration parameters
     */
    explicit TemporalConsistency(const Config& config);
    
    /**
     * @brief Destroy the Temporal Consistency object
     */
    ~TemporalConsistency();
    
    /**
     * @brief Initialize the temporal consistency module
     * 
     * @return true if initialization was successful
     */
    bool initialize();
    
    /**
     * @brief Process a frame for temporal consistency
     * 
     * @param current_frame The current frame to process
     * @param output_frame The output frame with temporal consistency applied
     * @return true if processing was successful
     */
    bool process(const cv::Mat& current_frame, cv::Mat& output_frame);
    
    /**
     * @brief Reset the frame buffer and state
     */
    void reset();
    
    /**
     * @brief Set the config parameters
     * 
     * @param config The new configuration to use
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
    std::deque<cv::Mat> m_frame_buffer;
    std::deque<cv::Mat> m_gray_buffer;
    std::deque<cv::Mat> m_flow_buffer;
    std::mutex m_buffer_mutex;
    
    /**
     * @brief Calculate optical flow between two frames
     * 
     * @param prev_frame Previous frame (grayscale)
     * @param curr_frame Current frame (grayscale)
     * @param flow Output flow field
     * @return true if successful
     */
    bool calculateOpticalFlow(const cv::Mat& prev_frame, const cv::Mat& curr_frame, cv::Mat& flow);
    
    /**
     * @brief Warp a frame according to flow field
     * 
     * @param frame Frame to warp
     * @param flow Flow field
     * @param warped_frame Output warped frame
     * @return true if successful
     */
    bool warpFrame(const cv::Mat& frame, const cv::Mat& flow, cv::Mat& warped_frame);
    
    /**
     * @brief Detect if there's a scene change between frames
     * 
     * @param prev_frame Previous frame (grayscale)
     * @param curr_frame Current frame (grayscale)
     * @return true if a scene change is detected
     */
    bool detectSceneChange(const cv::Mat& prev_frame, const cv::Mat& curr_frame);
    
    /**
     * @brief Calculate a reliability mask for the flow field
     * 
     * @param flow Flow field
     * @param mask Output reliability mask (0-1)
     */
    void calculateFlowReliabilityMask(const cv::Mat& flow, cv::Mat& mask);
    
    /**
     * @brief Blend frames with variable weights based on motion
     * 
     * @param frames Vector of frames to blend
     * @param reliability_masks Vector of reliability masks
     * @param output Output blended frame
     */
    void blendFrames(const std::vector<cv::Mat>& frames, 
                     const std::vector<cv::Mat>& reliability_masks, 
                     cv::Mat& output);
};