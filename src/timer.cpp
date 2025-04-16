#include "timer.h"
#include <iostream>
#include <numeric>
#include <algorithm>
#include <iomanip>

Timer::Timer() {
    // No initialization needed
}

void Timer::start(const std::string& event_name) {
    auto& event = events[event_name];
    event.start_time = std::chrono::high_resolution_clock::now();
    event.is_running = true;
}

void Timer::stop(const std::string& event_name) {
    auto it = events.find(event_name);
    if (it == events.end() || !it->second.is_running) {
        std::cerr << "Warning: Trying to stop timer for non-started event: " 
                  << event_name << std::endl;
        return;
    }
    
    auto& event = it->second;
    auto end_time = std::chrono::high_resolution_clock::now();
    DurationType duration = end_time - event.start_time;
    
    event.durations.push_back(duration.count());
    event.is_running = false;
}

double Timer::getDuration(const std::string& event_name) const {
    auto it = events.find(event_name);
    if (it == events.end() || it->second.durations.empty()) {
        return -1.0;
    }
    
    return it->second.durations.back();
}

double Timer::getAverageDuration(const std::string& event_name) const {
    auto it = events.find(event_name);
    if (it == events.end() || it->second.durations.empty()) {
        return -1.0;
    }
    
    const auto& durations = it->second.durations;
    return std::accumulate(durations.begin(), durations.end(), 0.0) / durations.size();
}

void Timer::reset() {
    events.clear();
}

void Timer::printStats() const {
    std::cout << "\n=== Timer Statistics ===\n";
    std::cout << std::setw(25) << "Event" << " | " 
              << std::setw(10) << "Last (ms)" << " | " 
              << std::setw(10) << "Avg (ms)" << " | " 
              << std::setw(10) << "Min (ms)" << " | " 
              << std::setw(10) << "Max (ms)" << " | " 
              << std::setw(10) << "Count" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& [name, event] : events) {
        if (event.durations.empty()) {
            continue;
        }
        
        double avg = std::accumulate(event.durations.begin(), event.durations.end(), 0.0) / 
                     event.durations.size();
        double min = *std::min_element(event.durations.begin(), event.durations.end());
        double max = *std::max_element(event.durations.begin(), event.durations.end());
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(25) << name << " | " 
                  << std::setw(10) << event.durations.back() << " | " 
                  << std::setw(10) << avg << " | " 
                  << std::setw(10) << min << " | " 
                  << std::setw(10) << max << " | " 
                  << std::setw(10) << event.durations.size() << std::endl;
    }
    std::cout << std::endl;
}