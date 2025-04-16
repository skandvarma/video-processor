#include "upscaler.h"
#include <iostream>

// Check if OpenCV was built with CUDA support
#ifdef WITH_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#endif

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
            case Upscaler::SUPER_RES:
                // Super resolution not available on CPU, fall back to bicubic
                interpolation = cv::INTER_CUBIC;
                break;
        }
        
        cv::resize(input, output, cv::Size(m_target_width, m_target_height), 0, 0, interpolation);
        return true;
    }
    
private:
    Upscaler::Algorithm m_algorithm;
    int m_target_width;
    int m_target_height;
};

#ifdef WITH_CUDA
// GPU implementation using CUDA
class GPUImpl : public UpscalerImpl {
public:
    GPUImpl(Upscaler::Algorithm algorithm, int target_width, int target_height)
        : m_algorithm(algorithm),
          m_target_width(target_width),
          m_target_height(target_height) {
        
        // Super resolution is not implemented in this version
        if (algorithm == Upscaler::SUPER_RES) {
            std::cout << "Super Resolution algorithm not available, falling back to bicubic" << std::endl;
        }
    }
    
    bool upscale(const cv::Mat& input, cv::Mat& output) override {
        if (input.empty()) {
            return false;
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
            case Upscaler::SUPER_RES:
                // Super resolution not implemented, use bicubic
                interpolation = cv::INTER_CUBIC;
                break;
        }
        
        cv::cuda::resize(d_input, d_output, cv::Size(m_target_width, m_target_height), 0, 0, interpolation);
        
        // Download result back to CPU
        d_output.download(output);
        return true;
    }
    
private:
    Upscaler::Algorithm m_algorithm;
    int m_target_width;
    int m_target_height;
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
    if (!m_initialized || !m_impl) {
        std::cerr << "Upscaler not initialized" << std::endl;
        return false;
    }
    
    if (input.empty()) {
        std::cerr << "Input frame is empty" << std::endl;
        return false;
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
            return "Super Resolution";
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