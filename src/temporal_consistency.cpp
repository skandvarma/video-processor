#include "temporal_consistency.h"
#include <iostream>

// Check if OpenCV was built with CUDA support
#ifdef WITH_CUDA
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif

TemporalConsistency::TemporalConsistency() 
    : m_initialized(false) {
}

TemporalConsistency::TemporalConsistency(const Config& config)
    : m_config(config), m_initialized(false) {
}

TemporalConsistency::~TemporalConsistency() {
}

bool TemporalConsistency::initialize() {
    reset();
    
    // Check if GPU is available when requested
    if (m_config.use_gpu) {
#ifdef WITH_CUDA
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
            std::cout << "CUDA requested for temporal consistency but not available. Using CPU fallback." << std::endl;
            m_config.use_gpu = false;
        } else {
            std::cout << "Using CUDA for temporal consistency processing." << std::endl;
        }
#else
        std::cout << "CUDA requested for temporal consistency but OpenCV was built without CUDA support. Using CPU fallback." << std::endl;
        m_config.use_gpu = false;
#endif
    }
    
    m_initialized = true;
    return true;
}

void TemporalConsistency::reset() {
    std::lock_guard<std::mutex> lock(m_buffer_mutex);
    m_frame_buffer.clear();
    m_gray_buffer.clear();
    m_flow_buffer.clear();
}

void TemporalConsistency::setConfig(const Config& config) {
    std::lock_guard<std::mutex> lock(m_buffer_mutex);
    m_config = config;
}

TemporalConsistency::Config TemporalConsistency::getConfig() const {
    return m_config;
}

bool TemporalConsistency::process(const cv::Mat& current_frame, cv::Mat& output_frame) {
    if (!m_initialized) {
        std::cerr << "Temporal consistency module not initialized" << std::endl;
        current_frame.copyTo(output_frame);
        return false;
    }
    
    if (current_frame.empty()) {
        std::cerr << "Empty input frame" << std::endl;
        return false;
    }
    
    // Convert current frame to grayscale for optical flow
    cv::Mat current_gray;
    cv::cvtColor(current_frame, current_gray, cv::COLOR_BGR2GRAY);
    
    std::lock_guard<std::mutex> lock(m_buffer_mutex);
    
    // If this is the first frame or if buffer is still filling
    if (m_frame_buffer.empty()) {
        // First frame, just store it and return
        m_frame_buffer.push_back(current_frame.clone());
        m_gray_buffer.push_back(current_gray.clone());
        current_frame.copyTo(output_frame);
        return true;
    }
    
    // Check for scene change
    bool scene_change = false;
    if (!m_gray_buffer.empty()) {
        scene_change = detectSceneChange(m_gray_buffer.back(), current_gray);
    }
    
    // Calculate optical flow between current and previous frame
    cv::Mat flow;
    if (!scene_change && calculateOpticalFlow(m_gray_buffer.back(), current_gray, flow)) {
        m_flow_buffer.push_back(flow.clone());
    } else {
        // If scene change or flow calculation failed, reset buffer
        if (scene_change) {
            std::cout << "Scene change detected, resetting temporal buffer" << std::endl;
            m_frame_buffer.clear();
            m_gray_buffer.clear();
            m_flow_buffer.clear();
        }
        
        // Add current frame to buffer and return it as output
        m_frame_buffer.push_back(current_frame.clone());
        m_gray_buffer.push_back(current_gray.clone());
        current_frame.copyTo(output_frame);
        return true;
    }
    
    // If we have at least one flow field and two frames, we can apply temporal consistency
    if (!m_flow_buffer.empty() && m_frame_buffer.size() >= 1) {
        // Create a vector to hold all frames for blending
        std::vector<cv::Mat> frames_to_blend;
        std::vector<cv::Mat> reliability_masks;
        
        // Add current frame as the first one
        frames_to_blend.push_back(current_frame.clone());
        // Add identity reliability mask for current frame (full weight)
        cv::Mat identity_mask = cv::Mat::ones(current_frame.size(), CV_32F);
        reliability_masks.push_back(identity_mask);
        
        // For each previous frame and flow field
        for (size_t i = 0; i < m_flow_buffer.size() && i < m_frame_buffer.size(); i++) {
            // Get frame and flow
            const cv::Mat& prev_frame = m_frame_buffer[m_frame_buffer.size() - 1 - i];
            const cv::Mat& flow_field = m_flow_buffer[m_flow_buffer.size() - 1 - i];
            
            // Warp previous frame to align with current frame
            cv::Mat warped_frame;
            if (warpFrame(prev_frame, flow_field, warped_frame)) {
                // Calculate reliability mask for this warped frame
                cv::Mat reliability_mask;
                calculateFlowReliabilityMask(flow_field, reliability_mask);
                
                // Add to blend arrays
                frames_to_blend.push_back(warped_frame);
                reliability_masks.push_back(reliability_mask);
            }
        }
        
        // Blend all frames using calculated reliability masks
        if (frames_to_blend.size() > 1) {
            blendFrames(frames_to_blend, reliability_masks, output_frame);
        } else {
            // If we only have the current frame, just return it
            current_frame.copyTo(output_frame);
        }
    } else {
        // Not enough data for temporal blending yet
        current_frame.copyTo(output_frame);
    }
    
    // Update frame buffer
    m_frame_buffer.push_back(current_frame.clone());
    m_gray_buffer.push_back(current_gray.clone());
    
    // Keep buffer size in check
    while (m_frame_buffer.size() > m_config.buffer_size) {
        m_frame_buffer.pop_front();
        m_gray_buffer.pop_front();
    }
    
    while (m_flow_buffer.size() >= m_config.buffer_size) {
        m_flow_buffer.pop_front();
    }
    
    return true;
}

bool TemporalConsistency::calculateOpticalFlow(const cv::Mat& prev_frame, const cv::Mat& curr_frame, cv::Mat& flow) {
    try {
        if (m_config.use_gpu) {
#ifdef WITH_CUDA
            // GPU implementation using CUDA
            cv::cuda::GpuMat d_prev, d_curr, d_flow;
            d_prev.upload(prev_frame);
            d_curr.upload(curr_frame);
            
            // Create Farneback flow object with configured parameters
            auto farneback = cv::cuda::FarnebackOpticalFlow::create(
                m_config.levels,
                m_config.pyr_scale,
                false, // fastPyramids
                m_config.winsize,
                m_config.iterations,
                m_config.poly_n,
                m_config.poly_sigma,
                m_config.flags
            );
            
            // Calculate flow
            farneback->calc(d_prev, d_curr, d_flow);
            
            // Download result
            d_flow.download(flow);
#else
            // Fall back to CPU if CUDA was requested but not available at compile time
            cv::calcOpticalFlowFarneback(
                prev_frame, curr_frame, flow,
                m_config.pyr_scale, m_config.levels, m_config.winsize,
                m_config.iterations, m_config.poly_n, m_config.poly_sigma,
                m_config.flags
            );
#endif
        } else {
            // CPU implementation
            cv::calcOpticalFlowFarneback(
                prev_frame, curr_frame, flow,
                m_config.pyr_scale, m_config.levels, m_config.winsize,
                m_config.iterations, m_config.poly_n, m_config.poly_sigma,
                m_config.flags
            );
        }
        
        return !flow.empty();
    } catch (const cv::Exception& e) {
        std::cerr << "Error calculating optical flow: " << e.what() << std::endl;
        return false;
    }
}

bool TemporalConsistency::warpFrame(const cv::Mat& frame, const cv::Mat& flow, cv::Mat& warped_frame) {
    try {
        if (m_config.use_gpu) {
#ifdef WITH_CUDA
            // GPU implementation
            cv::cuda::GpuMat d_frame, d_flow, d_warped;
            d_frame.upload(frame);
            d_flow.upload(flow);
            
            // Remap using the flow field
            cv::cuda::remap(d_frame, d_warped, flow, cv::noArray(), 
                          cv::INTER_LINEAR, cv::BORDER_REPLICATE);
            
            // Download result
            d_warped.download(warped_frame);
#else
            // CPU implementation if CUDA requested but not available
            // Create map from flow field
            cv::Mat map_x(flow.size(), CV_32F);
            cv::Mat map_y(flow.size(), CV_32F);
            
            for (int y = 0; y < flow.rows; y++) {
                for (int x = 0; x < flow.cols; x++) {
                    const cv::Vec2f& f = flow.at<cv::Vec2f>(y, x);
                    map_x.at<float>(y, x) = x + f[0];
                    map_y.at<float>(y, x) = y + f[1];
                }
            }
            
            // Apply mapping
            cv::remap(frame, warped_frame, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
#endif
        } else {
            // CPU implementation
            // Create map from flow field
            cv::Mat map_x(flow.size(), CV_32F);
            cv::Mat map_y(flow.size(), CV_32F);
            
            for (int y = 0; y < flow.rows; y++) {
                for (int x = 0; x < flow.cols; x++) {
                    const cv::Vec2f& f = flow.at<cv::Vec2f>(y, x);
                    map_x.at<float>(y, x) = x + f[0];
                    map_y.at<float>(y, x) = y + f[1];
                }
            }
            
            // Apply mapping
            cv::remap(frame, warped_frame, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
        }
        
        return !warped_frame.empty();
    } catch (const cv::Exception& e) {
        std::cerr << "Error warping frame: " << e.what() << std::endl;
        return false;
    }
}

bool TemporalConsistency::detectSceneChange(const cv::Mat& prev_frame, const cv::Mat& curr_frame) {
    try {
        // Calculate normalized histogram difference
        cv::Mat prev_hist, curr_hist;
        int histSize = 64;
        float range[] = {0, 256};
        const float* histRange = {range};
        
        cv::calcHist(&prev_frame, 1, 0, cv::Mat(), prev_hist, 1, &histSize, &histRange);
        cv::calcHist(&curr_frame, 1, 0, cv::Mat(), curr_hist, 1, &histSize, &histRange);
        
        cv::normalize(prev_hist, prev_hist, 0, 1, cv::NORM_MINMAX);
        cv::normalize(curr_hist, curr_hist, 0, 1, cv::NORM_MINMAX);
        
        // Calculate histogram difference using correlation method
        double correlation = cv::compareHist(prev_hist, curr_hist, cv::HISTCMP_CORREL);
        double difference = 1.0 - correlation; // Convert correlation to difference (0-1)
        
        // Also calculate mean absolute difference between frames
        cv::Mat diff;
        cv::absdiff(prev_frame, curr_frame, diff);
        double mad = cv::mean(diff)[0];
        
        // Combine histogram difference and mean absolute difference with weights
        double combined_diff = (difference * 100.0) + (mad * 0.5);
        
        // Detect scene change if combined difference exceeds threshold
        bool is_scene_change = combined_diff > m_config.scene_change_threshold;
        
        if (is_scene_change) {
            std::cout << "Scene change detected: diff=" << combined_diff 
                      << " (hist_diff=" << difference * 100.0 
                      << ", mad=" << mad << ")" << std::endl;
        }
        
        return is_scene_change;
    } catch (const cv::Exception& e) {
        std::cerr << "Error detecting scene change: " << e.what() << std::endl;
        return false;
    }
}

void TemporalConsistency::calculateFlowReliabilityMask(const cv::Mat& flow, cv::Mat& mask) {
    try {
        // Create a mask based on flow magnitude
        mask = cv::Mat(flow.size(), CV_32F, cv::Scalar(1.0));
        
        // For each pixel, calculate flow magnitude and check against threshold
        for (int y = 0; y < flow.rows; y++) {
            for (int x = 0; x < flow.cols; x++) {
                const cv::Vec2f& f = flow.at<cv::Vec2f>(y, x);
                float magnitude = std::sqrt(f[0] * f[0] + f[1] * f[1]);
                
                // If magnitude is greater than threshold, reduce reliability
                // using a sigmoid-like falloff for smooth transition
                if (magnitude > m_config.motion_threshold) {
                    float reliability = std::exp(-(magnitude - m_config.motion_threshold) / 10.0f);
                    reliability = std::max(0.0f, std::min(1.0f, reliability));
                    mask.at<float>(y, x) = reliability;
                }
            }
        }
        
        // Apply Gaussian blur to the mask for smoother transitions
        cv::GaussianBlur(mask, mask, cv::Size(15, 15), 5.0);
    } catch (const cv::Exception& e) {
        std::cerr << "Error calculating flow reliability mask: " << e.what() << std::endl;
        // In case of error, create a neutral mask
        mask = cv::Mat(flow.size(), CV_32F, cv::Scalar(0.5));
    }
}

void TemporalConsistency::blendFrames(const std::vector<cv::Mat>& frames, 
                                      const std::vector<cv::Mat>& reliability_masks, 
                                      cv::Mat& output) {
    try {
        if (frames.empty() || frames[0].empty()) {
            if (!frames.empty()) {
                frames[0].copyTo(output);
            }
            return;
        }
        
        // Create result frame initialized with zeros
        output = cv::Mat::zeros(frames[0].size(), frames[0].type());
        
        // Create a weight accumulation buffer
        cv::Mat weight_sum = cv::Mat::zeros(frames[0].size(), CV_32F);
        
        // Prepare frame weights with exponential decay for older frames
        std::vector<float> frame_weights(frames.size());
        for (size_t i = 0; i < frames.size(); i++) {
            // Exponential decay weight - newer frames have higher weight
            frame_weights[i] = std::exp(-static_cast<float>(i) / 2.0f);
        }
        
        // Apply blending with variable weights
        for (size_t i = 0; i < frames.size(); i++) {
            // Skip invalid frames
            if (frames[i].empty() || frames[i].size() != frames[0].size()) {
                continue;
            }
            
            // Use reliability mask if available, otherwise use all ones
            cv::Mat mask;
            if (i < reliability_masks.size() && !reliability_masks[i].empty()) {
                mask = reliability_masks[i];
            } else {
                mask = cv::Mat::ones(frames[i].size(), CV_32F);
            }
            
            // Apply temporal blend strength parameter to all but the current frame
            float blend_factor = (i == 0) ? 1.0f : m_config.blend_strength;
            
            // Convert frame to float for blending
            cv::Mat frame_float;
            frames[i].convertTo(frame_float, CV_32F);
            
            // For each pixel, accumulate weighted value and weight sum
            for (int y = 0; y < output.rows; y++) {
                for (int x = 0; x < output.cols; x++) {
                    // Calculate pixel weight using reliability mask and frame weight
                    float weight = frame_weights[i] * mask.at<float>(y, x) * blend_factor;
                    
                    // Accumulate weighted pixel value
                    cv::Vec3f& out_pixel = output.at<cv::Vec3f>(y, x);
                    const cv::Vec3b& in_pixel = frames[i].at<cv::Vec3b>(y, x);
                    
                    out_pixel[0] += weight * in_pixel[0];
                    out_pixel[1] += weight * in_pixel[1];
                    out_pixel[2] += weight * in_pixel[2];
                    
                    // Accumulate weight
                    weight_sum.at<float>(y, x) += weight;
                }
            }
        }
        
        // Normalize by weight sum
        for (int y = 0; y < output.rows; y++) {
            for (int x = 0; x < output.cols; x++) {
                float weight = weight_sum.at<float>(y, x);
                if (weight > 0) {
                    cv::Vec3f& pixel = output.at<cv::Vec3f>(y, x);
                    pixel /= weight;
                } else {
                    // If weight is zero, use the current frame
                    output.at<cv::Vec3f>(y, x) = cv::Vec3f(frames[0].at<cv::Vec3b>(y, x));
                }
            }
        }
        
        // Convert back to 8-bit
        output.convertTo(output, CV_8UC3);
    } catch (const cv::Exception& e) {
        std::cerr << "Error blending frames: " << e.what() << std::endl;
        // In case of error, return the current frame
        if (!frames.empty()) {
            frames[0].copyTo(output);
        }
    }
}