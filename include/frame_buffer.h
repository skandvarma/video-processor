#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <memory>

/**
 * @brief A thread-safe ring buffer for video frames
 * 
 * This class implements a zero-copy ring buffer for efficiently passing
 * frames between producer and consumer threads. It uses cv::Mat with
 * reference counting to avoid unnecessary copying of frame data.
 */
class FrameBuffer {
public:
    /**
     * @brief Construct a new FrameBuffer with specified capacity
     * @param capacity Maximum number of frames the buffer can hold
     */
    explicit FrameBuffer(size_t capacity = 10);
    
    /**
     * @brief Add a frame to the buffer (producer)
     * 
     * This function is called by the producer thread to add a new frame
     * to the buffer. If the buffer is full, it will overwrite the oldest
     * frame.
     * 
     * @param frame The frame to add to the buffer
     * @param blocking If true, blocks until space is available
     * @return true if frame was added, false if buffer was full (when non-blocking)
     */
    bool pushFrame(const cv::Mat& frame, bool blocking = true);
    
    /**
     * @brief Get the next frame from the buffer (consumer)
     * 
     * This function is called by the consumer thread to retrieve the 
     * next frame from the buffer.
     * 
     * @param frame Output parameter that will contain the next frame
     * @param blocking If true, will block until a frame is available
     * @return true if a frame was retrieved, false if buffer is empty (when non-blocking)
     */
    bool popFrame(cv::Mat& frame, bool blocking = true);
    
    /**
     * @brief Get the number of frames currently in the buffer
     * @return The number of frames
     */
    size_t size() const;
    
    /**
     * @brief Check if the buffer is empty
     * @return true if the buffer is empty, false otherwise
     */
    bool empty() const;
    
    /**
     * @brief Check if the buffer is full
     * @return true if the buffer is full, false otherwise
     */
    bool full() const;
    
    /**
     * @brief Get the buffer capacity
     * @return The maximum number of frames the buffer can hold
     */
    size_t capacity() const;
    
    /**
     * @brief Clear all frames from the buffer
     */
    void clear();
    
private:
    size_t m_capacity;                 // Maximum number of frames
    std::vector<cv::Mat> m_frames;     // The actual frame storage
    size_t m_head;                     // Index where next frame will be written
    size_t m_tail;                     // Index where next frame will be read
    std::atomic<size_t> m_size;        // Number of frames currently in buffer
    
    mutable std::mutex m_mutex;        // Mutex for thread safety
    std::condition_variable m_not_empty; // Signaled when buffer becomes non-empty
    std::condition_variable m_not_full;  // Signaled when buffer becomes non-full
    
    // Advance index with wrap-around
    size_t nextIndex(size_t current) const;
};