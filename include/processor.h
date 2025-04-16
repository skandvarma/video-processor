#pragma once

#include <opencv2/opencv.hpp>
#include <functional>
#include <string>
#include <memory>
#include <vector>
#include <mutex>

/**
 * @brief Video frame processor for applying transformations to frames
 * 
 * This class handles various video processing operations that can be applied
 * to frames in the pipeline. It supports chaining multiple operations and
 * GPU acceleration when available.
 */
class Processor {
public:
    // Type definition for a processing function
    using ProcessFunction = std::function<void(const cv::Mat&, cv::Mat&)>;

    /**
     * @brief Structure representing a processing operation
     */
    struct Operation {
        std::string name;
        ProcessFunction func;
        bool enabled;
        
        Operation(const std::string& n, ProcessFunction f)
            : name(n), func(f), enabled(true) {}
    };
    
    // Forward declaration of implementation class
    class ProcessorImpl;

    /**
     * @brief Construct a new Processor object
     * @param use_gpu Whether to use GPU acceleration if available
     */
    explicit Processor(bool use_gpu = true);
    
    /**
     * @brief Destroy the Processor object
     */
    ~Processor();
    
    /**
     * @brief Initialize the processor
     * @return true if initialization was successful
     */
    bool initialize();
    
    /**
     * @brief Process a frame with all registered operations
     * @param input Input frame
     * @param output Output frame
     * @return true if processing was successful
     */
    bool process(const cv::Mat& input, cv::Mat& output);
    
    /**
     * @brief Add a processing operation to the pipeline
     * @param name Name of the operation
     * @param func Processing function
     * @return Reference to this processor for chaining
     */
    Processor& addOperation(const std::string& name, ProcessFunction func);
    
    /**
     * @brief Add common pre-processing operations
     * @return Reference to this processor for chaining
     */
    Processor& addDefaultPreProcessing();
    
    /**
     * @brief Add common post-processing operations
     * @return Reference to this processor for chaining
     */
    Processor& addDefaultPostProcessing();
    
    /**
     * @brief Enable or disable a specific operation
     * @param name Name of the operation
     * @param enabled Whether the operation should be enabled
     */
    void enableOperation(const std::string& name, bool enabled);
    
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
     * @brief Check if GPU acceleration is available
     * @return true if GPU acceleration is available
     */
    static bool isGPUAvailable();
    
    /**
     * @brief Get duration of last processing operation in milliseconds
     * @return Processing time in milliseconds
     */
    double getLastProcessingTime() const;
    
private:
    std::vector<Operation> m_operations;
    bool m_use_gpu;
    bool m_initialized;
    mutable std::mutex m_mutex;
    double m_last_processing_time;
    
    // Internal implementation class (to handle GPU/CPU differences)
    std::unique_ptr<ProcessorImpl> m_impl;
};