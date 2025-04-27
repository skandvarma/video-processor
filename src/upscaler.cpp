#include "upscaler.h"
#include <iostream>

// Check if OpenCV was built with CUDA support
#ifdef WITH_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#endif

#include "dnn_super_res.h"


// Forward declaration of implementation classes
class UpscalerImpl;
class CPUImpl;
#ifdef WITH_CUDA
class GPUImpl;
#endif

// Base implementation class
class UpscalerImpl {
public:
    virtual ~UpscalerImpl() = default;
    virtual bool upscale(const cv::Mat& input, cv::Mat& output) = 0;
};

// CPU implementation
class CPUImpl : public UpscalerImpl {
public:
    CPUImpl(Upscaler::Algorithm algorithm, int target_width, int target_height)
        : m_algorithm(algorithm),
          m_target_width(target_width),
          m_target_height(target_height) {
    }
    
    bool upscale(const cv::Mat& input, cv::Mat& output) override {
        if (input.empty()) {
            return false;
        }
        
        // For super-res algorithm, use multi-stage upscaling with enhancements
        if (m_algorithm == Upscaler::SUPER_RES) {
            return upscaleSuperRes(input, output);
        }
        
        int interpolation = cv::INTER_LINEAR; // Default
        
        // Map algorithm enum to OpenCV interpolation constant
        switch (m_algorithm) {
            case Upscaler::NEAREST:
                interpolation = cv::INTER_NEAREST;
                break;
            case Upscaler::BILINEAR:
                interpolation = cv::INTER_LINEAR;
                break;
            case Upscaler::BICUBIC:
                interpolation = cv::INTER_CUBIC;
                break;
            case Upscaler::LANCZOS:
                interpolation = cv::INTER_LANCZOS4;
                break;
            default:
                interpolation = cv::INTER_LANCZOS4;
                break;
        }
        
        cv::resize(input, output, cv::Size(m_target_width, m_target_height), 0, 0, interpolation);
        
        // Apply post-processing for better visual quality (except for NEAREST which is meant to be pixelated)
        if (m_algorithm != Upscaler::NEAREST) {
            enhanceDetails(output);
        }
        
        return true;
    }
    
private:
    Upscaler::Algorithm m_algorithm;
    int m_target_width;
    int m_target_height;
    
    // Enhanced multi-stage upscaling for SUPER_RES algorithm
    bool upscaleSuperRes(const cv::Mat& input, cv::Mat& output) {
        // Step 1: Preserve colors by converting to YCrCb
        cv::Mat ycrcb;
        cv::cvtColor(input, ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        
        // Step 2: Apply bilateral filter to Y channel to reduce noise while preserving edges
        cv::Mat y_filtered;
        cv::bilateralFilter(channels[0], y_filtered, 5, 50, 50);
        
        // Step 3: Upscale Y channel (luminance) with Lanczos for best detail
        cv::Mat y_upscaled;
        cv::resize(y_filtered, y_upscaled, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_LANCZOS4);
        
        // Step 4: Enhance details in the Y channel
        cv::Mat y_enhanced;
        enhanceDetailsY(y_upscaled, y_enhanced);
        
        // Step 5: Upscale chroma channels with bilinear (less critical for quality)
        cv::Mat cr_upscaled, cb_upscaled;
        cv::resize(channels[1], cr_upscaled, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_LINEAR);
        cv::resize(channels[2], cb_upscaled, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_LINEAR);
        
        // Step 6: Merge channels back
        std::vector<cv::Mat> upscaled_channels = {y_enhanced, cr_upscaled, cb_upscaled};
        cv::Mat merged;
        cv::merge(upscaled_channels, merged);
        
        // Step 7: Convert back to BGR
        cv::cvtColor(merged, output, cv::COLOR_YCrCb2BGR);
 
        // Step 8: Final color enhancement for YouTube-like quality
        enhanceColors(output);
        
        return true;
    }
    
    // Enhance details in BGR image
    void enhanceDetails(cv::Mat& image) {
        // Apply a subtle sharpening filter
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            -0.1, -0.1, -0.1,
            -0.1,  1.8, -0.1,
            -0.1, -0.1, -0.1);
            
        cv::filter2D(image, image, -1, kernel);
    }
    
    // Enhance details in Y channel (more aggressive since it's just luminance)
    void enhanceDetailsY(const cv::Mat& input, cv::Mat& output) {
        // Create a sharpening kernel
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            -1, -1, -1,
            -1,  9, -1,
            -1, -1, -1);
            
        cv::filter2D(input, output, -1, kernel);
    }
    
    // Enhance colors for more vibrant output (YouTube-like)
    void enhanceColors(cv::Mat& image) {
        // Convert to Lab color space for better color manipulation
        cv::Mat lab;
        cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
        
        // Split into channels
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);
        
        // Increase a and b channels slightly for more vibrant colors
        channels[1] = channels[1] * 1.1; // Slightly increase 'a' (controls green-red)
        channels[2] = channels[2] * 1.1; // Slightly increase 'b' (controls blue-yellow)
        
        // Merge channels
        cv::merge(channels, lab);
        
        // Convert back to BGR
        cv::cvtColor(lab, image, cv::COLOR_Lab2BGR);
        
        // Slightly increase contrast
        image.convertTo(image, -1, 1.05, 0);
    }
};

#ifdef WITH_CUDA
// GPU implementation using CUDA
class GPUImpl : public UpscalerImpl {
public:
    GPUImpl(Upscaler::Algorithm algorithm, int target_width, int target_height)
        : m_algorithm(algorithm),
          m_target_width(target_width),
          m_target_height(target_height) {
    }
    
    bool upscale(const cv::Mat& input, cv::Mat& output) override {
        if (input.empty()) {
            return false;
        }
        
        // For super-res algorithm, use multi-stage processing with CUDA
        if (m_algorithm == Upscaler::SUPER_RES) {
            return upscaleSuperRes(input, output);
        }
        
        // Upload input to GPU
        cv::cuda::GpuMat d_input;
        d_input.upload(input);
        
        // GPU Mat for output
        cv::cuda::GpuMat d_output;
        
        // Use standard resize with appropriate interpolation
        int interpolation = cv::INTER_LINEAR; // Default
        
        switch (m_algorithm) {
            case Upscaler::NEAREST:
                interpolation = cv::INTER_NEAREST;
                break;
            case Upscaler::BILINEAR:
                interpolation = cv::INTER_LINEAR;
                break;
            case Upscaler::BICUBIC:
                interpolation = cv::INTER_CUBIC;
                break;
            case Upscaler::LANCZOS:
                interpolation = cv::INTER_LANCZOS4;
                break;
            default:
                interpolation = cv::INTER_LANCZOS4;
                break;
        }
        
        cv::cuda::resize(d_input, d_output, cv::Size(m_target_width, m_target_height), 0, 0, interpolation);
        
        // Apply post-processing for better quality (except for NEAREST)
        if (m_algorithm != Upscaler::NEAREST) {
            // Download for post-processing (sharpening isn't always available in CUDA)
            d_output.download(output);
            enhanceDetails(output);
        } else {
            // Just download the result without enhancement
            d_output.download(output);
        }
        
        return true;
    }
    
private:
    Upscaler::Algorithm m_algorithm;
    int m_target_width;
    int m_target_height;
    
    // CUDA-accelerated multi-stage upscaling
    bool upscaleSuperRes(const cv::Mat& input, cv::Mat& output) {
        try {
            // Step 1: Convert to YCrCb for better processing
            cv::Mat ycrcb;
            cv::cvtColor(input, ycrcb, cv::COLOR_BGR2YCrCb);
            std::vector<cv::Mat> channels;
            cv::split(ycrcb, channels);
            
            // Step 2: For filtering, we'll do it on CPU since your CUDA build may not have bilateral filter
            cv::Mat y_filtered;
            cv::bilateralFilter(channels[0], y_filtered, 5, 50, 50);
            
            // Step 3: Upload filtered Y channel to GPU
            cv::cuda::GpuMat d_y_filtered, d_y_upscaled;
            d_y_filtered.upload(y_filtered);
            
            // Step 4: Upscale Y channel with CUBIC instead of LANCZOS4 which isn't supported by your CUDA build
            cv::cuda::resize(d_y_filtered, d_y_upscaled, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_CUBIC);
            
            // Step 5: Upscale chroma channels (bilinear is fine for chroma)
            cv::cuda::GpuMat d_cr, d_cb, d_cr_upscaled, d_cb_upscaled;
            d_cr.upload(channels[1]);
            d_cb.upload(channels[2]);
            cv::cuda::resize(d_cr, d_cr_upscaled, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_LINEAR);
            cv::cuda::resize(d_cb, d_cb_upscaled, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_LINEAR);
            
            // Step 6: Download channels from GPU
            cv::Mat y_upscaled, cr_upscaled, cb_upscaled;
            d_y_upscaled.download(y_upscaled);
            d_cr_upscaled.download(cr_upscaled);
            d_cb_upscaled.download(cb_upscaled);
            
            // Step 7: Enhance details in Y channel (add a bit more sharpening to compensate for not using Lanczos)
            cv::Mat y_enhanced;
            enhanceDetailsY(y_upscaled, y_enhanced);
            
            // Step 8: Merge channels
            std::vector<cv::Mat> upscaled_channels = {y_enhanced, cr_upscaled, cb_upscaled};
            cv::Mat merged;
            cv::merge(upscaled_channels, merged);
            
            // Step 9: Convert back to BGR
            cv::cvtColor(merged, output, cv::COLOR_YCrCb2BGR);
            
            // Extra step: Reduce noise in dark areas
            reduceDarkAreaNoise(output);

            
            // Step 10: Final color enhancement
            enhanceColors(output);
            
            return true;
        }
        catch (const cv::Exception& e) {
            std::cerr << "CUDA error in super resolution: " << e.what() << std::endl;
            
            // Fallback to standard CUBIC resize (not Lanczos)
            cv::cuda::GpuMat d_input, d_output;
            d_input.upload(input);
            try {
                cv::cuda::resize(d_input, d_output, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_CUBIC);
                d_output.download(output);
                
                // Still apply some enhancement
                enhanceDetails(output);
            }
            catch (const cv::Exception& e2) {
                std::cerr << "CUDA resize failed again: " << e2.what() << ", falling back to CPU" << std::endl;
                
                // Ultimate fallback to CPU
                cv::resize(input, output, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_CUBIC);
                enhanceDetails(output);
            }
            
            return true;
        }
    }
    
    // Enhance details in BGR image
    void enhanceDetails(cv::Mat& image) {
        // Apply a subtle sharpening filter
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            -0.1, -0.1, -0.1,
            -0.1,  1.8, -0.1,
            -0.1, -0.1, -0.1);
            
        cv::filter2D(image, image, -1, kernel);
    }
    
    // Enhance details in Y channel (more aggressive since it's just luminance)
    void enhanceDetailsY(const cv::Mat& input, cv::Mat& output) {
        // Create a more refined sharpening kernel for the luminance channel
        float kernel_data[] = {
            -0.5f, -0.5f, -0.5f,
            -0.5f,  5.0f, -0.5f,
            -0.5f, -0.5f, -0.5f
        };
        cv::Mat kernel = cv::Mat(3, 3, CV_32F, kernel_data);
                    
        // Apply filter
        cv::filter2D(input, output, -1, kernel);
        
        // Blend with original for a more natural look
        cv::addWeighted(input, 0.3, output, 0.7, 0, output);
    }

    void reduceDarkAreaNoise(cv::Mat& image) {
        // Convert to YCrCb for better luminance isolation
        cv::Mat ycrcb;
        cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
        
        // Split channels
        std::vector<cv::Mat> channels;
        cv::split(ycrcb, channels);
        
        // Create a mask for dark areas
        cv::Mat darkMask;
        cv::threshold(channels[0], darkMask, 60, 1.0, cv::THRESH_BINARY_INV);
        
        // Apply stronger bilateral filter only to dark areas
        cv::Mat filtered;
        cv::bilateralFilter(channels[0], filtered, 5, 30, 30);
        
        // Blend original and filtered based on mask
        channels[0] = channels[0].mul(1.0 - darkMask) + filtered.mul(darkMask);
        
        // Merge channels
        cv::merge(channels, ycrcb);
        
        // Convert back to BGR
        cv::cvtColor(ycrcb, image, cv::COLOR_YCrCb2BGR);
    }
    
    // Enhance colors for more vibrant output (YouTube-like)
    void enhanceColors(cv::Mat& image) {
        // Convert to Lab color space for better color manipulation
        cv::Mat lab;
        cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
        
        // Split into channels
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);
        
        // More subtle color enhancement (reduce from 1.1 to 1.05)
        channels[1] = channels[1] * 1.05; // Slightly increase 'a' (controls green-red)
        channels[2] = channels[2] * 1.05; // Slightly increase 'b' (controls blue-yellow)
        
        // Merge channels
        cv::merge(channels, lab);
        
        // Convert back to BGR
        cv::cvtColor(lab, image, cv::COLOR_Lab2BGR);
        
        // Slightly increase contrast but less aggressively
        image.convertTo(image, -1, 1.03, 0);
        
        // Add a very subtle brightness boost
        cv::Mat brightened;
        image.convertTo(brightened, -1, 1.0, 3);  // Add a small constant value
        
        // Blend the original with brightened for more natural look
        double alpha = 0.7; // Blend factor - 70% original, 30% brightened
        cv::addWeighted(image, alpha, brightened, 1-alpha, 0, image);
    }
};
#endif

// Upscaler implementation
Upscaler::Upscaler(Algorithm algorithm, bool use_gpu)
    : m_algorithm(algorithm),
      m_use_gpu(use_gpu),
      m_initialized(false),
      m_target_width(0),
      m_target_height(0),
      m_impl(nullptr) {
    
    // Adjust if GPU requested but not available
    if (m_use_gpu && !isGPUAvailable()) {
        std::cout << "GPU acceleration requested but not available. Using CPU implementation." << std::endl;
        m_use_gpu = false;
    }
}

Upscaler::~Upscaler() = default;

bool Upscaler::initialize(int target_width, int target_height) {
    if (target_width <= 0 || target_height <= 0) {
        std::cerr << "Invalid target resolution: " << target_width << "x" << target_height << std::endl;
        return false;
    }
    
    m_target_width = target_width;
    m_target_height = target_height;
    
    return initializeImpl();
}

bool Upscaler::upscale(const cv::Mat& input, cv::Mat& output) {
    if (input.empty()) {
        std::cerr << "Input frame is empty" << std::endl;
        return false;
    }
    
    // Handle DNN super-resolution if it's initialized
    if (m_algorithm == SUPER_RES && m_dnn_sr && m_dnn_sr->isInitialized()) {
        return m_dnn_sr->upscale(input, output);
    }
    
    // Check if the standard implementation is initialized
    if (!m_impl) {
        std::cerr << "Upscaler not initialized" << std::endl;
        
        // Fall back to a simple resize if all else fails
        try {
            cv::resize(input, output, cv::Size(m_target_width, m_target_height), 0, 0, cv::INTER_LINEAR);
            return true;
        } catch (const cv::Exception& e) {
            std::cerr << "Basic resize failed: " << e.what() << std::endl;
            return false;
        }
    }
    
    return m_impl->upscale(input, output);
}

void Upscaler::setAlgorithm(Algorithm algorithm) {
    if (m_algorithm != algorithm) {
        m_algorithm = algorithm;
        if (m_initialized) {
            initializeImpl();
        }
    }
}

bool Upscaler::setUseGPU(bool use_gpu) {
    // Check if GPU is available when requested
    if (use_gpu && !isGPUAvailable()) {
        std::cerr << "GPU acceleration requested but not available" << std::endl;
        return false;
    }
    
    if (m_use_gpu != use_gpu) {
        m_use_gpu = use_gpu;
        if (m_initialized) {
            initializeImpl();
        }
    }
    
    return true;
}

bool Upscaler::isUsingGPU() const {
    return m_use_gpu;
}

std::string Upscaler::getAlgorithmName() const {
    switch (m_algorithm) {
        case NEAREST:
            return "Nearest Neighbor";
        case BILINEAR:
            return "Bilinear";
        case BICUBIC:
            return "Bicubic";
        case LANCZOS:
            return "Lanczos";
        case SUPER_RES:
            return "YouTube-Quality";
        default:
            return "Unknown";
    }
}

bool Upscaler::isGPUAvailable() {
#ifdef WITH_CUDA
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
#else
    return false;
#endif
}

bool Upscaler::initializeImpl() {
    // Clean up existing implementation
    m_impl.reset();
    
    if (m_target_width <= 0 || m_target_height <= 0) {
        std::cerr << "Invalid target resolution" << std::endl;
        m_initialized = false;
        return false;
    }

    if (m_algorithm == SUPER_RES) {
        try {
            m_dnn_sr = std::make_unique<DnnSuperRes>();
            m_dnn_sr->setTargetSize(m_target_width, m_target_height);
            m_dnn_sr->setUseGPU(m_use_gpu);
            
            if (m_dnn_sr->initialize()) {
                std::cout << "Using DNN Super Resolution for upscaling" << std::endl;
                m_initialized = true;
                return true;
            } else {
                std::cerr << "Failed to initialize DNN Super Resolution, falling back to standard method" << std::endl;
                m_dnn_sr.reset();
                // Continue with standard implementation
            }
        } catch (const std::exception& e) {
            std::cerr << "Error creating DNN Super Resolution: " << e.what() << std::endl;
            m_dnn_sr.reset();
            // Continue with standard implementation
        }
    }
    
#ifdef WITH_CUDA
    if (m_use_gpu) {
        try {
            m_impl = std::make_unique<GPUImpl>(m_algorithm, m_target_width, m_target_height);
            std::cout << "Using GPU-accelerated upscaling with " << getAlgorithmName() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to initialize GPU implementation: " << e.what() << std::endl;
            m_use_gpu = false; // Fall back to CPU
        }
    }
#endif
    
    // Create CPU implementation if GPU is not being used
    if (!m_impl) {
        m_impl = std::make_unique<CPUImpl>(m_algorithm, m_target_width, m_target_height);
        std::cout << "Using CPU upscaling with " << getAlgorithmName() << std::endl;
    }
    
    m_initialized = (m_impl != nullptr);
    return m_initialized;
}