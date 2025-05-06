#include "frame_buffer.h"
#include <iostream>

FrameBuffer::FrameBuffer(size_t capacity)
    : m_capacity(capacity),
      m_head(0),
      m_tail(0),
      m_size(0) {
    // Pre-allocate frames with empty Mats
    m_frames.resize(capacity);
}

bool FrameBuffer::pushFrame(const cv::Mat& frame, bool blocking) {
    if (frame.empty()) {
        std::cerr << "Warning: Attempting to push empty frame to buffer" << std::endl;
        return false;
    }
    
    std::unique_lock<std::mutex> lock(m_mutex);
    
    if (full()) {
        if (!blocking) {
            return false;
        }
        
        // Wait until buffer has space
        m_not_full.wait(lock, [this]() { return !full(); });
    }
    
    // If the buffer is getting full, optimize memory usage
    if (m_size > m_capacity * 0.8) {
        // Use existing memory when possible to reduce allocations
        frame.copyTo(m_frames[m_head], cv::noArray());
    } else {
        // Otherwise use standard copy
        frame.copyTo(m_frames[m_head]);
    }
    
    // Update head position
    m_head = nextIndex(m_head);
    m_size++;
    
    // Notify consumers that buffer is not empty
    m_not_empty.notify_one();
    
    return true;
}

bool FrameBuffer::popFrame(cv::Mat& frame, bool blocking) {
    std::unique_lock<std::mutex> lock(m_mutex);
    
    if (empty()) {
        if (!blocking) {
            return false;
        }
        
        // Wait until buffer has data
        m_not_empty.wait(lock, [this]() { return !empty(); });
    }
    
    // Get frame at tail (zero-copy assignment using cv::Mat reference counting)
    frame = m_frames[m_tail];
    
    // Set the buffer slot to empty to allow its memory to be reclaimed if needed
    m_frames[m_tail] = cv::Mat();
    
    // Update tail position
    m_tail = nextIndex(m_tail);
    m_size--;
    
    // Notify producers that buffer is not full
    m_not_full.notify_one();
    
    return true;
}

size_t FrameBuffer::size() const {
    return m_size.load();
}

bool FrameBuffer::empty() const {
    return size() == 0;
}

bool FrameBuffer::full() const {
    return size() >= m_capacity;
}

size_t FrameBuffer::capacity() const {
    return m_capacity;
}

void FrameBuffer::clear() {
    std::unique_lock<std::mutex> lock(m_mutex);
    
    // Clear all frames
    for (auto& frame : m_frames) {
        frame.release();
    }
    
    m_head = 0;
    m_tail = 0;
    m_size = 0;
    
    // Notify producers that buffer is not full
    m_not_full.notify_all();
}

size_t FrameBuffer::nextIndex(size_t current) const {
    return (current + 1) % m_capacity;
}