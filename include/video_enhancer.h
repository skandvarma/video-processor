#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

class VideoEnhancer {
public:
    enum EnhancementLevel {
        NONE,
        LIGHT,
        MEDIUM,
        STRONG,
        YOUTUBE
    };
    
    VideoEnhancer(EnhancementLevel level = MEDIUM);
    ~VideoEnhancer();
    
    bool initialize();
    bool enhance(const cv::Mat& input, cv::Mat& output);
    void setLevel(EnhancementLevel level);
    EnhancementLevel getLevel() const;
    
private:
    EnhancementLevel m_level;
    bool m_initialized;
    
    // Color processing
    void enhanceColors(cv::Mat& image);
    void adjustContrast(cv::Mat& image, float factor);
    void adjustSaturation(cv::Mat& image, float factor);
    void adjustGamma(cv::Mat& image, float gamma);

    // LUT-based color grading methods
    cv::Mat createCinematicLUT();
    void applyLUT(cv::Mat& image, const cv::Mat& lut3D);
    void addVignette(cv::Mat& image, float strength);
    bool loadCubeLUT(const std::string& filepath, cv::Mat& lut3D);
    
    // Detail processing
    void enhanceDetails(cv::Mat& image);
    void sharpenAdaptive(cv::Mat& image, float strength);
    void reduceNoise(cv::Mat& image, float strength);
    
    // Advanced techniques
    void localContrastEnhancement(cv::Mat& image);
    void colorGrading(cv::Mat& image); // For cinematic look
    void enhanceDarkAreas(cv::Mat& image);
    void recoverHighlights(cv::Mat& image);
};