#include "video_enhancer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

// If M_PI is not defined in your environment:
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

VideoEnhancer::VideoEnhancer(EnhancementLevel level)
    : m_level(level), m_initialized(false) {
}

VideoEnhancer::~VideoEnhancer() {
}

bool VideoEnhancer::initialize() {
    m_initialized = true;
    return true;
}

bool VideoEnhancer::enhance(const cv::Mat& input, cv::Mat& output) {
    if (!m_initialized) {
        std::cerr << "VideoEnhancer not initialized" << std::endl;
        return false;
    }
    
    if (input.empty()) {
        std::cerr << "Input frame is empty" << std::endl;
        return false;
    }
    
    // Skip processing if level is NONE
    if (m_level == NONE) {
        input.copyTo(output);
        return true;
    }
    
    // Create a working copy
    input.copyTo(output);
    
    // Process based on enhancement level
    switch (m_level) {
        case LIGHT:
            reduceNoise(output, 0.3f);
            enhanceDetails(output);
            enhanceColors(output);
            break;
            
        case MEDIUM:
            reduceNoise(output, 0.5f);
            enhanceDetails(output);
            enhanceColors(output);
            adjustContrast(output, 1.05f);
            break;
            
        case STRONG:
            reduceNoise(output, 0.6f);
            enhanceDetails(output);
            enhanceColors(output);
            adjustContrast(output, 1.1f);
            localContrastEnhancement(output);
            break;
            
        case YOUTUBE:
            reduceNoise(output, 0.5f);
            enhanceDarkAreas(output);
            enhanceDetails(output);
            colorGrading(output);
            localContrastEnhancement(output);
            recoverHighlights(output);
            break;
            
        default:
            break;
    }
    
    return true;
}

void VideoEnhancer::setLevel(EnhancementLevel level) {
    m_level = level;
}

VideoEnhancer::EnhancementLevel VideoEnhancer::getLevel() const {
    return m_level;
}

void VideoEnhancer::enhanceColors(cv::Mat& image) {
    // Convert to LAB colorspace for better color processing
    cv::Mat lab;
    cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
    
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(lab, channels);
    
    // Enhance a and b channels for more vivid colors
    float factor = (m_level == YOUTUBE) ? 1.08f : 1.05f;
    channels[1] = channels[1] * factor;
    channels[2] = channels[2] * factor;
    
    // Merge and convert back
    cv::merge(channels, lab);
    cv::cvtColor(lab, image, cv::COLOR_Lab2BGR);
}

void VideoEnhancer::adjustContrast(cv::Mat& image, float factor) {
    cv::Mat result;
    image.convertTo(result, -1, factor, 0);
    result.copyTo(image);
}

void VideoEnhancer::adjustSaturation(cv::Mat& image, float factor) {
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);
    
    channels[1] = channels[1] * factor;
    
    cv::merge(channels, hsv);
    cv::cvtColor(hsv, image, cv::COLOR_HSV2BGR);
}

void VideoEnhancer::adjustGamma(cv::Mat& image, float gamma) {
    cv::Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i)
        p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    
    cv::Mat result;
    cv::LUT(image, lookUpTable, result);
    result.copyTo(image);
}

void VideoEnhancer::enhanceDetails(cv::Mat& image) {
    // Convert to YCrCb for better detail processing
    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    
    // Apply sharpening to Y channel only
    cv::Mat y = channels[0];
    cv::Mat sharpened;
    
    // Create sharpening kernel
    float strength = (m_level == YOUTUBE) ? 0.8f : 0.5f;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
        -0.1f*strength, -0.1f*strength, -0.1f*strength,
        -0.1f*strength,  1.0f + 0.8f*strength, -0.1f*strength,
        -0.1f*strength, -0.1f*strength, -0.1f*strength);
    
    cv::filter2D(y, sharpened, -1, kernel);
    
    // Blend with original for more natural look
    cv::addWeighted(y, 0.3, sharpened, 0.7, 0, channels[0]);
    
    // Merge channels
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, image, cv::COLOR_YCrCb2BGR);
}

void VideoEnhancer::sharpenAdaptive(cv::Mat& image, float strength) {
    // Convert to grayscale to detect edges
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    
    // Detect edges using Laplacian
    cv::Mat edges;
    cv::Laplacian(gray, edges, CV_8U, 3);
    
    // Threshold edges to create mask
    cv::Mat mask;
    cv::threshold(edges, mask, 25, 255, cv::THRESH_BINARY);
    
    // Dilate mask to extend edges
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(mask, mask, kernel);
    
    // Convert mask to float 0-1
    mask.convertTo(mask, CV_32F, 1.0/255.0);
    
    // Apply sharpening filter to entire image
    cv::Mat sharpened;
    cv::Mat sharpen_kernel = (cv::Mat_<float>(3, 3) << 
        0, -strength, 0,
        -strength, 1+4*strength, -strength,
        0, -strength, 0);
    
    cv::filter2D(image, sharpened, -1, sharpen_kernel);
    
    // Blend based on mask
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float m = mask.at<float>(i, j);
            if (m > 0) {
                image.at<cv::Vec3b>(i, j) = sharpened.at<cv::Vec3b>(i, j);
            }
        }
    }
}

void VideoEnhancer::reduceNoise(cv::Mat& image, float strength) {
    // Skip if image is too small for bilateral filter
    if (image.rows < 5 || image.cols < 5)
        return;
    
    // Adjust filter parameters based on strength
    int d = 5;
    double sigmaColor = 15 * strength;
    double sigmaSpace = 15 * strength;
    
    cv::Mat result;
    cv::bilateralFilter(image, result, d, sigmaColor, sigmaSpace);
    result.copyTo(image);
}

void VideoEnhancer::localContrastEnhancement(cv::Mat& image) {
    // Convert to YCrCb
    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    
    // Apply CLAHE to the Y channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(2.0);
    clahe->setTilesGridSize(cv::Size(8, 8));
    
    cv::Mat enhanced;
    clahe->apply(channels[0], enhanced);
    
    // Blend with original Y channel for more natural look
    cv::addWeighted(channels[0], 0.3, enhanced, 0.7, 0, channels[0]);
    
    // Merge and convert back
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, image, cv::COLOR_YCrCb2BGR);
}

void VideoEnhancer::colorGrading(cv::Mat& image) {
    // LUT-based color grading implementation
    // This simulates professional color grading used in film/YouTube
    
    // If we haven't loaded the LUT yet, create it
    static cv::Mat lut3D;
    static bool lut_initialized = false;
    
    if (!lut_initialized) {
        // Create a 3D LUT with 33x33x33 size (standard for color grading)
        lut3D = createCinematicLUT();
        lut_initialized = true;
    }
    
    // Apply the 3D LUT to the image
    applyLUT(image, lut3D);
    
    // Optional: Add a subtle vignette for cinematic effect
    addVignette(image, 0.3f);
}

cv::Mat VideoEnhancer::createCinematicLUT() {
    // Create a 3D LUT with 33x33x33 size (standard for color grading)
    const int LUT_SIZE = 33;
    cv::Mat lut(LUT_SIZE * LUT_SIZE, LUT_SIZE, CV_8UC3);
    
    // Step 1: Fill with identity transformation (no change)
    for (int b = 0; b < LUT_SIZE; b++) {
        for (int g = 0; g < LUT_SIZE; g++) {
            for (int r = 0; r < LUT_SIZE; r++) {
                // Map from LUT coordinates to BGR color values (0-255)
                uchar bVal = cv::saturate_cast<uchar>((b * 255) / (LUT_SIZE - 1));
                uchar gVal = cv::saturate_cast<uchar>((g * 255) / (LUT_SIZE - 1));
                uchar rVal = cv::saturate_cast<uchar>((r * 255) / (LUT_SIZE - 1));
                
                // Store in the LUT
                cv::Vec3b& color = lut.at<cv::Vec3b>(b + g * LUT_SIZE, r);
                color[0] = bVal;
                color[1] = gVal;
                color[2] = rVal;
            }
        }
    }
    
    // Step 2: Apply professional-style adjustments to the LUT
    
    // 2.1: Create cinematic contrast curve (S-curve)
    // This gives rich shadows and bright highlights
    for (int i = 0; i < lut.rows; i++) {
        for (int j = 0; j < lut.cols; j++) {
            cv::Vec3b& pixel = lut.at<cv::Vec3b>(i, j);
            
            // Apply different curves to each channel
            for (int c = 0; c < 3; c++) {
                float val = pixel[c] / 255.0f;
                
                // Apply S-curve for contrast
                // y = 0.5 - sin(π * (x - 0.5)) / (2π)
                val = 0.5f - sin(M_PI * (val - 0.5f)) / (2.0f * M_PI);
                
                // Adjust gamma slightly
                val = powf(val, 0.95f); 
                
                // Store back
                pixel[c] = cv::saturate_cast<uchar>(val * 255.0f);
            }
        }
    }
    
    // 2.2: Adjust color balance - slightly warm highlights, cool shadows (orange & teal)
    for (int i = 0; i < lut.rows; i++) {
        for (int j = 0; j < lut.cols; j++) {
            cv::Vec3b& pixel = lut.at<cv::Vec3b>(i, j);
            
            // Calculate brightness (simplified)
            float brightness = (pixel[0] + pixel[1] + pixel[2]) / (3.0f * 255.0f);
            
            // Adjust based on brightness
            if (brightness > 0.6f) { // Highlights - warm up
                // Increase red, slightly decrease blue
                float factor = (brightness - 0.6f) * 2.5f;
                pixel[2] = cv::saturate_cast<uchar>(pixel[2] * (1.0f + 0.07f * factor)); // Red
                pixel[0] = cv::saturate_cast<uchar>(pixel[0] * (1.0f - 0.05f * factor)); // Blue
            } 
            else if (brightness < 0.4f) { // Shadows - cool down
                // Increase blue, decrease red
                float factor = (0.4f - brightness) * 2.5f;
                pixel[0] = cv::saturate_cast<uchar>(pixel[0] * (1.0f + 0.07f * factor)); // Blue
                pixel[2] = cv::saturate_cast<uchar>(pixel[2] * (1.0f - 0.05f * factor)); // Red
            }
        }
    }
    
    // 2.3: Add subtle color shifts - vibrance and "film" look
    for (int i = 0; i < lut.rows; i++) {
        for (int j = 0; j < lut.cols; j++) {
            cv::Vec3b& pixel = lut.at<cv::Vec3b>(i, j);
            
            // Convert to HSV-like space for easier color manipulation
            float b = pixel[0] / 255.0f;
            float g = pixel[1] / 255.0f;
            float r = pixel[2] / 255.0f;
            
            // Find max and min values
            float max_val = std::max(std::max(r, g), b);
            float min_val = std::min(std::min(r, g), b);
            float chroma = max_val - min_val;
            
            // Only adjust saturated colors
            if (chroma > 0.1f) {
                // Boost specific color ranges (cinematic style)
                // Teals and cyans
                if (b > g && b > r) {
                    b = std::min(1.0f, b * 1.05f);
                    g = std::min(1.0f, g * 1.03f);
                }
                // Oranges and skin tones
                else if (r > g && g > b) {
                    r = std::min(1.0f, r * 1.06f);
                    g = std::min(1.0f, g * 1.02f);
                }
                // Greens (slightly muted, film-like)
                else if (g > r && g > b) {
                    g = g * 0.97f;
                }
            }
            
            // Convert back
            pixel[0] = cv::saturate_cast<uchar>(b * 255.0f);
            pixel[1] = cv::saturate_cast<uchar>(g * 255.0f);
            pixel[2] = cv::saturate_cast<uchar>(r * 255.0f);
        }
    }
    
    // 2.4: Final adjustment - lift shadows slightly for YouTube style
    for (int i = 0; i < lut.rows; i++) {
        for (int j = 0; j < lut.cols; j++) {
            cv::Vec3b& pixel = lut.at<cv::Vec3b>(i, j);
            
            // Calculate brightness
            float brightness = (pixel[0] + pixel[1] + pixel[2]) / (3.0f * 255.0f);
            
            // Lift shadows
            if (brightness < 0.3f) {
                float lift = 10.0f * (0.3f - brightness);
                pixel[0] = cv::saturate_cast<uchar>(pixel[0] + lift);
                pixel[1] = cv::saturate_cast<uchar>(pixel[1] + lift);
                pixel[2] = cv::saturate_cast<uchar>(pixel[2] + lift);
            }
        }
    }
    
    return lut;
}

void VideoEnhancer::applyLUT(cv::Mat& image, const cv::Mat& lut3D) {
    const int LUT_SIZE = 33; // Must match LUT creation
    
    // Process each pixel
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b& pixel = image.at<cv::Vec3b>(i, j);
            
            // Get normalized coordinates in the LUT
            float bVal = pixel[0] / 255.0f * (LUT_SIZE - 1);
            float gVal = pixel[1] / 255.0f * (LUT_SIZE - 1);
            float rVal = pixel[2] / 255.0f * (LUT_SIZE - 1);
            
            // Get the 8 surrounding points in the LUT cube
            int b0 = std::floor(bVal);
            int g0 = std::floor(gVal);
            int r0 = std::floor(rVal);
            
            int b1 = std::min(b0 + 1, LUT_SIZE - 1);
            int g1 = std::min(g0 + 1, LUT_SIZE - 1);
            int r1 = std::min(r0 + 1, LUT_SIZE - 1);
            
            // Get fractional parts for interpolation
            float bFrac = bVal - b0;
            float gFrac = gVal - g0;
            float rFrac = rVal - r0;
            
            // 3D linear interpolation (trilinear)
            // First interpolate in B direction
            cv::Vec3b c000 = lut.at<cv::Vec3b>(b0 + g0 * LUT_SIZE, r0);
            cv::Vec3b c100 = lut.at<cv::Vec3b>(b1 + g0 * LUT_SIZE, r0);
            cv::Vec3b c00 = c000 * (1 - bFrac) + c100 * bFrac;
            
            cv::Vec3b c010 = lut.at<cv::Vec3b>(b0 + g1 * LUT_SIZE, r0);
            cv::Vec3b c110 = lut.at<cv::Vec3b>(b1 + g1 * LUT_SIZE, r0);
            cv::Vec3b c01 = c010 * (1 - bFrac) + c110 * bFrac;
            
            cv::Vec3b c001 = lut.at<cv::Vec3b>(b0 + g0 * LUT_SIZE, r1);
            cv::Vec3b c101 = lut.at<cv::Vec3b>(b1 + g0 * LUT_SIZE, r1);
            cv::Vec3b c10 = c001 * (1 - bFrac) + c101 * bFrac;
            
            cv::Vec3b c011 = lut.at<cv::Vec3b>(b0 + g1 * LUT_SIZE, r1);
            cv::Vec3b c111 = lut.at<cv::Vec3b>(b1 + g1 * LUT_SIZE, r1);
            cv::Vec3b c11 = c011 * (1 - bFrac) + c111 * bFrac;
            
            // Interpolate in G direction
            cv::Vec3b c0 = c00 * (1 - gFrac) + c01 * gFrac;
            cv::Vec3b c1 = c10 * (1 - gFrac) + c11 * gFrac;
            
            // Finally, interpolate in R direction
            cv::Vec3b result = c0 * (1 - rFrac) + c1 * rFrac;
            
            // Update the pixel
            pixel = result;
        }
    }
}

void VideoEnhancer::addVignette(cv::Mat& image, float strength) {
    int borderSize = image.cols / 15;
    cv::Mat mask = cv::Mat::ones(image.size(), CV_32F);
    cv::rectangle(mask, 
                 cv::Rect(borderSize, borderSize, 
                          image.cols - 2*borderSize, image.rows - 2*borderSize),
                 cv::Scalar(0), -1);
    cv::GaussianBlur(mask, mask, cv::Size(borderSize*2+1, borderSize*2+1), 0);
    mask = 1.0 - mask;
    
    // Apply vignette
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            float m = 1.0 - (mask.at<float>(i, j) * strength);
            image.at<cv::Vec3b>(i, j) = cv::Vec3b(
                cv::saturate_cast<uchar>(image.at<cv::Vec3b>(i, j)[0] * m),
                cv::saturate_cast<uchar>(image.at<cv::Vec3b>(i, j)[1] * m),
                cv::saturate_cast<uchar>(image.at<cv::Vec3b>(i, j)[2] * m)
            );
        }
    }
}

// Add this to load industry-standard LUTs in CUBE format
bool VideoEnhancer::loadCubeLUT(const std::string& filepath, cv::Mat& lut3D) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open LUT file: " << filepath << std::endl;
        return false;
    }
    
    int size = 0;
    std::string line;
    while (std::getline(file, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;
            
        // Get LUT size
        if (line.find("LUT_3D_SIZE") != std::string::npos) {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> size;
            continue;
        }
        
        // Process data points if size is known
        if (size > 0) {
            // Check if line contains 3 float values
            float r, g, b;
            std::istringstream iss(line);
            if (iss >> r >> g >> b) {
                // Convert 0-1 range to 0-255
                uchar rVal = cv::saturate_cast<uchar>(r * 255.0f);
                uchar gVal = cv::saturate_cast<uchar>(g * 255.0f);
                uchar bVal = cv::saturate_cast<uchar>(b * 255.0f);
                
                // Calculate position in the LUT
                int rIdx = (r * (size - 1) + 0.5f);
                int gIdx = (g * (size - 1) + 0.5f);
                int bIdx = (b * (size - 1) + 0.5f);
                
                // Store in LUT
                lut3D.at<cv::Vec3b>(bIdx + gIdx * size, rIdx) = cv::Vec3b(bVal, gVal, rVal);
            }
        }
    }
    
    file.close();
    return true;
}

void VideoEnhancer::enhanceDarkAreas(cv::Mat& image) {
    // Convert to YCrCb
    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    
    // Create a mask for dark areas
    cv::Mat darkMask;
    cv::threshold(channels[0], darkMask, 60, 1.0, cv::THRESH_BINARY_INV);
    cv::GaussianBlur(darkMask, darkMask, cv::Size(5, 5), 0);
    
    // Lighten dark areas
    cv::Mat y = channels[0];
    cv::Mat y_light;
    y.convertTo(y_light, -1, 1.0, 15);
    
    // Blend based on mask
    channels[0] = y.mul(1.0 - darkMask) + y_light.mul(darkMask);
    
    // Merge channels
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, image, cv::COLOR_YCrCb2BGR);
}

void VideoEnhancer::recoverHighlights(cv::Mat& image) {
    // Convert to YCrCb
    cv::Mat ycrcb;
    cv::cvtColor(image, ycrcb, cv::COLOR_BGR2YCrCb);
    
    // Split channels
    std::vector<cv::Mat> channels;
    cv::split(ycrcb, channels);
    
    // Create a mask for highlight areas
    cv::Mat highlightMask;
    cv::threshold(channels[0], highlightMask, 235, 1.0, cv::THRESH_BINARY);
    cv::GaussianBlur(highlightMask, highlightMask, cv::Size(5, 5), 0);
    
    // Recover highlights by reducing brightness
    cv::Mat y = channels[0];
    cv::Mat y_recovered;
    y.convertTo(y_recovered, -1, 0.9, 0);
    
    // Blend based on mask
    channels[0] = y.mul(1.0 - highlightMask) + y_recovered.mul(highlightMask);
    
    // Merge channels
    cv::merge(channels, ycrcb);
    cv::cvtColor(ycrcb, image, cv::COLOR_YCrCb2BGR);
}