## Multimodal Gesture Recognition

A real-time computer vision solution for simultaneous recognition of hand gestures, facial expressions, and body postures using OpenCV and MediaPipe.

> 🚧 Active Development: This project is evolving. Core features are stable, but expect frequent enhancements.

### Overview

The project implements a comprehensive gesture recognition pipeline that processes multiple input modalities in real-time:
- Hand gesture tracking and classification
- Facial expression analysis
- Body posture detection
- Multimodal gesture fusion

### Key Features

- Multi-threaded processing for optimal performance
- Configurable gesture detection thresholds
- Low-latency real-time feedback
- Modular architecture for easy extension
- Multimodal gesture fusion for complex interactions

### Current Support

**Hand Gestures**
- Peace sign ✌️
- OK gesture 👌

**Facial Expressions**
- Smile detection 😊
- Eyebrow movement tracking 🤨

**Body Postures**
- Shrug gesture 🤷

**Multimodal Gestures**
- Thinking pose 🤔 (combines raised eyebrows and hand-to-chin positioning)

### Requirements

- Python 3.9+
- Webcam
- OpenCV compatible GPU (optional, improves performance)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-gesture-recognition.git
cd multimodal-gesture-recognition

# Install dependencies
pip3 install -r requirements.txt

# Run the application
python3 gesture_recognition.py
```

### Gesture Details

#### Basic Gestures
- **Peace Sign** ✌️: Detection of raised index and middle fingers
- **OK Gesture** 👌: Recognition of thumb and index finger forming a circle
- **Smile** 😊: Analysis of mouth corners and width-to-height ratio
- **Shrug** 🤷: Tracking of shoulder elevation relative to neutral position

#### Multimodal Gestures
- **Thinking Pose** 🤔
  - Combines multiple modalities:
    1. Facial tracking: Raised eyebrows detection
    2. Hand tracking: Hand position near chin
  - Uses adaptive thresholds for natural variation
  - Implements temporal smoothing for stable detection

### Usage

The application will launch in fullscreen mode. Use the following controls:
- `q` - Quit the application
- `r` - Reset eyebrow tracking baseline
- `esc` - Exit fullscreen

#### On-Screen Information
- Real-time gesture recognition status
- Available gesture list
- Visual feedback for detected features

#### Roadmap

- [ ] Additional multimodal gestures
- [ ] Gesture combination sequences
- [ ] Custom gesture training interface
- [ ] Performance optimizations
- [ ] Configuration UI
- [ ] Advanced gesture analytics

### Technical Details

#### Multimodal Integration
- Real-time fusion of multiple input streams
- Adaptive threshold management
- Temporal smoothing for stable detection
- State management for complex gestures

#### Performance Features
- Efficient landmark processing
- Optimized detection pipelines
- Smart gesture state management

#### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file in the root directory of this repository for detailed terms and conditions.

---
*Built with OpenCV and MediaPipe*