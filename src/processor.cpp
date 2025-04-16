#include "processor.h"
#include <iostream>
#include <chrono>

// Check if OpenCV was built with CUDA support
#ifdef WITH_CUDA
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#endif

// Implementation class definition
class Processor::ProcessorImpl {
public:
    virtual ~ProcessorImpl() = default;
    virtual bool initialize() = 0;
    virtual void process(const cv::Mat& input, cv::Mat& output, 
                         const std::vector<Processor::Operation>& operations,
                         double& processing_time) = 0;
};

// CPU implementation
class CPUProcessorImpl : public Processor::ProcessorImpl {
public:
    CPUProcessorImpl() {}
    
    bool initialize() override {
        return true;
    }
    
    void process(const cv::Mat& input, cv::Mat& output, 
                 const std::vector<Processor::Operation>& operations,
                 double& processing_time) override {
        
        if (input.empty()) {
            return;
        }
        
        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Make a copy of the input frame as the starting point
        input.copyTo(output);
        
        // Apply each enabled operation in sequence
        cv::Mat temp;
        for (const auto& op : operations) {
            if (op.enabled) {
                op.func(output, temp);
                output = temp;
            }
        }
        
        // Calculate processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};

#ifdef WITH_CUDA
// GPU implementation using CUDA
class GPUProcessorImpl : public Processor::ProcessorImpl {
public:
    GPUProcessorImpl() {}
    
    bool initialize() override {
        // Check if CUDA is available
        if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
            std::cerr << "No CUDA devices found" << std::endl;
            return false;
        }
        
        // Set the current device (using the first one)
        cv::cuda::setDevice(0);
        
        // Get device info
        cv::cuda::DeviceInfo device_info;
        std::cout << "Using GPU: " << device_info.name() 
                  << " (compute capability: " << device_info.majorVersion() 
                  << "." << device_info.minorVersion() << ")" << std::endl;
        
        return true;
    }
    
    void process(const cv::Mat& input, cv::Mat& output, 
                 const std::vector<Processor::Operation>& operations,
                 double& processing_time) override {
        
        if (input.empty()) {
            return;
        }
        
        // Start timing
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Upload input to GPU
        cv::cuda::GpuMat d_input;
        d_input.upload(input);
        
        // Apply each enabled operation
        // Note: For GPU processing, ideally each operation would be implemented
        // to work directly with GpuMat to avoid unnecessary transfers
        cv::cuda::GpuMat d_output = d_input;
        cv::cuda::GpuMat d_temp;
        cv::Mat h_temp;
        
        for (const auto& op : operations) {
            if (!op.enabled) {
                continue;
            }
            
            // For this implementation, we'll download to CPU, apply the operation, 
            // and upload back to GPU. In a real implementation, operations should
            // be GPU-native to avoid this overhead.
            d_output.download(h_temp);
            cv::Mat h_result;
            op.func(h_temp, h_result);
            d_temp.upload(h_result);
            d_output = d_temp;
        }
        
        // Download result back to CPU
        d_output.download(output);
        
        // Calculate processing time
        auto end_time = std::chrono::high_resolution_clock::now();
        processing_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    }
};
#endif

// Processor implementation
Processor::Processor(bool use_gpu)
    : m_use_gpu(use_gpu),
      m_initialized(false),
      m_last_processing_time(0.0) {
    
    // Adjust if GPU requested but not available
    if (m_use_gpu && !isGPUAvailable()) {
        std::cout << "GPU acceleration requested but not available. Using CPU implementation." << std::endl;
        m_use_gpu = false;
    }
    
    // Create appropriate implementation
    if (m_use_gpu) {
#ifdef WITH_CUDA
        m_impl = std::make_unique<GPUProcessorImpl>();
#else
        m_impl = std::make_unique<CPUProcessorImpl>();
#endif
    } else {
        m_impl = std::make_unique<CPUProcessorImpl>();
    }
}

Processor::~Processor() = default;

bool Processor::initialize() {
    if (m_initialized) {
        return true;
    }
    
    if (!m_impl) {
        std::cerr << "Processor implementation not created" << std::endl;
        return false;
    }
    
    m_initialized = m_impl->initialize();
    return m_initialized;
}

bool Processor::process(const cv::Mat& input, cv::Mat& output) {
    if (!m_initialized || !m_impl) {
        std::cerr << "Processor not initialized" << std::endl;
        return false;
    }
    
    if (input.empty()) {
        std::cerr << "Input frame is empty" << std::endl;
        return false;
    }
    
    // Lock to ensure operations list isn't modified during processing
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // Process the frame using the implementation
    double processing_time = 0.0;
    m_impl->process(input, output, m_operations, processing_time);
    m_last_processing_time = processing_time;
    
    return true;
}

Processor& Processor::addOperation(const std::string& name, ProcessFunction func) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_operations.emplace_back(name, func);
    return *this;
}

Processor& Processor::addDefaultPreProcessing() {
    // Add common pre-processing operations
    
    // Noise reduction (Gaussian blur)
    addOperation("denoise", [](const cv::Mat& input, cv::Mat& output) {
        cv::GaussianBlur(input, output, cv::Size(5, 5), 0);
    });
    
    // Color correction
    addOperation("color_correction", [](const cv::Mat& input, cv::Mat& output) {
        cv::cvtColor(input, output, cv::COLOR_BGR2YUV);
        std::vector<cv::Mat> channels;
        cv::split(output, channels);
        
        // Apply histogram equalization to the Y channel
        cv::equalizeHist(channels[0], channels[0]);
        
        cv::merge(channels, output);
        cv::cvtColor(output, output, cv::COLOR_YUV2BGR);
    });
    
    return *this;
}

Processor& Processor::addDefaultPostProcessing() {
    // Add common post-processing operations
    
    // Sharpen
    addOperation("sharpen", [](const cv::Mat& input, cv::Mat& output) {
        cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
            -1, -1, -1,
            -1,  9, -1,
            -1, -1, -1);
        cv::filter2D(input, output, -1, kernel);
    });
    
    // Contrast enhancement
    addOperation("contrast", [](const cv::Mat& input, cv::Mat& output) {
        input.convertTo(output, -1, 1.2, 10);
    });
    
    return *this;
}

void Processor::enableOperation(const std::string& name, bool enabled) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    for (auto& op : m_operations) {
        if (op.name == name) {
            op.enabled = enabled;
            return;
        }
    }
    
    std::cerr << "Operation not found: " << name << std::endl;
}

bool Processor::setUseGPU(bool use_gpu) {
    // Check if GPU is available when requested
    if (use_gpu && !isGPUAvailable()) {
        std::cerr << "GPU acceleration requested but not available" << std::endl;
        return false;
    }
    
    if (m_use_gpu != use_gpu) {
        m_use_gpu = use_gpu;
        
        // Recreate implementation
        if (m_use_gpu) {
#ifdef WITH_CUDA
            m_impl = std::make_unique<GPUProcessorImpl>();
#else
            m_impl = std::make_unique<CPUProcessorImpl>();
#endif
        } else {
            m_impl = std::make_unique<CPUProcessorImpl>();
        }
        
        // Re-initialize
        m_initialized = false;
        initialize();
    }
    
    return true;
}

bool Processor::isUsingGPU() const {
    return m_use_gpu;
}

bool Processor::isGPUAvailable() {
#ifdef WITH_CUDA
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
#else
    return false;
#endif
}

double Processor::getLastProcessingTime() const {
    return m_last_processing_time;
}