
---

# 🚀 Low-Latency Video Processing System

A high-performance C++ and OpenCV-based system for **real-time video capture, processing, and display** with **low latency** and **upscaling capabilities**.

---

## 📌 Project Overview

This project provides a modular, multi-threaded framework to:

* Capture live camera feeds or video files
* Process frames using high-quality upscaling (bicubic or super-resolution)
* Display and optionally record video with minimal delay

Built for speed, modularity, and extendability — optimized for **real-time performance**.

---

## ✨ Features

* 🔍 **Camera detection & video file support**
* ⚙️ **Thread-safe zero-copy frame buffer** (Producer–Consumer model)
* ⚡ **Multi-threaded architecture** (Capture / Process / Display)
* 🚀 **GPU acceleration** with CUDA (optional)
* 🎞️ **Multiple upscaling algorithms**:

  * **Bicubic** (fast, default)
  * **Super-Resolution** (neural net, highest quality)
* 💡 **Temporal smoothing** for FPS boost with visual consistency
* 📷 **Screenshot capture**
* 🎥 **Real-time video recording** with multiple formats
* 📊 **Performance stats & diagnostics**

---

## 🛠️ System Requirements

* C++17-compatible compiler
* CMake ≥ 3.10
* OpenCV 4.x (with CUDA support for GPU acceleration)
* \[Optional] CUDA Toolkit

---

## 🧱 Build Instructions

```bash
# Clone repository
git clone https://github.com/yourusername/video-processor.git
cd video-processor

# Create and enter build directory
mkdir build && cd build

# Configure and build
cmake ..
make

# Run the application
../bin/video_processor
```

---

## 🚦 Usage

```bash
./bin/video_processor [camera_index | video_file_path] [options]
```

### CLI Options:

| Option                 | Description                                       |
| ---------------------- | ------------------------------------------------- |
| `--output`, `-o`       | Output file path                                  |
| `--record`, `-r`       | Start recording immediately                       |
| `--super-res`, `-sr`   | Use neural network super-resolution               |
| `--format`, `-fmt`     | Output format: `mp4`, `h264`, `yuv`, `avi`, `mkv` |
| `--resolution`, `-res` | Set output resolution: `width height`             |
| `--fast`, `-f`         | Process as fast as possible (ignore frame rate)   |

---

## 📂 Example Commands

```bash
# Live webcam feed with bicubic (default)
./bin/video_processor

# Webcam with super-resolution
./bin/video_processor --super-res

# Process video file with super-res and save output
./bin/video_processor video.mp4 --super-res --output output.mp4 --record

# Record with specific format
./bin/video_processor video.mp4 --output output.mp4 --format h264 --record

# Custom resolution
./bin/video_processor --resolution 1280 720
```

---

## ⌨️ Keyboard Controls (Live Mode)

| Key | Action               |
| --- | -------------------- |
| `q` | Quit the application |
| `r` | Toggle recording     |
| `s` | Save a screenshot    |

---

## 🔧 Performance Tips

* ✅ **Use bicubic** for live video (best real-time quality/performance balance)
* 🔍 **Use super-resolution** for post-processing (high quality, more compute)
* 🎛️ Increase buffer size for smoother super-res (but adds latency)
* 📉 Lower input resolution to speed up processing
* ⚡ **Enable GPU** (OpenCV with CUDA) for significant performance gain

---

## 🧬 Architecture

The system uses a 3-thread pipeline:

```
[Capture Thread] → [Processing Thread] → [Display Thread]
       ↓                    ↓                    ↓
   Raw Buffer         Processed Buffer       Output / Record
```

This design ensures non-blocking operation and smooth real-time performance.

---

## 💾 Output Formats

| Format | Notes                               |
| ------ | ----------------------------------- |
| MP4    | Default, widely supported           |
| H.264  | Efficient, high quality             |
| YUV    | Raw format, large files             |
| AVI    | Uses MJPG codec                     |
| MKV    | Uses X264 codec, flexible container |

> Format support may depend on your OpenCV build and installed codecs.

---

## 🛠 Troubleshooting

| Issue            | Fix                                                        |
| ---------------- | ---------------------------------------------------------- |
| Dropped frames   | Increase buffer, reduce resolution, or use bicubic         |
| Codec errors     | Try different format or check OpenCV codec support         |
| Slow performance | Enable CUDA, reduce resolution, or avoid super-res in live |

---

## 📃 License

[MIT License](LICENSE)

---

