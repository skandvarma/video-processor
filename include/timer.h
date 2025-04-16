#pragma once

#include <chrono>
#include <string>
#include <vector>
#include <map>

class Timer {
public:
    Timer();
    
    // Start timing a specific event
    void start(const std::string& event_name);
    
    // Stop timing an event and record duration
    void stop(const std::string& event_name);
    
    // Get duration of an event in milliseconds
    double getDuration(const std::string& event_name) const;
    
    // Get average duration over multiple calls
    double getAverageDuration(const std::string& event_name) const;
    
    // Reset all timings
    void reset();
    
    // Print timing statistics
    void printStats() const;
    
private:
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    using DurationType = std::chrono::duration<double, std::milli>;
    
    struct EventTiming {
        TimePoint start_time;
        bool is_running;
        std::vector<double> durations;
        
        EventTiming() : is_running(false) {}
    };
    
    std::map<std::string, EventTiming> events;
};