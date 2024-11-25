## Multimodal Gesture Recognition
A real-time computer vision solution for simultaneous recognition of hand gestures, facial expressions, and body postures using OpenCV and MediaPipe.

> 🚧 Active Development: This project is evolving. Core features are stable, but expect frequent enhancements.

### Overview
The project implements a comprehensive gesture recognition pipeline that processes multiple input modalities in real-time:
- Hand gesture tracking and classification
- Facial expression analysis
- Body posture detection
- Multimodal gesture fusion
- Rhythmic movement detection

### Key Features
- Multi-threaded processing for optimal performance
- Configurable gesture detection thresholds
- Low-latency real-time feedback
- Modular architecture for easy extension
- Multimodal gesture fusion for complex interactions
- Cultural gesture recognition capabilities

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

**Rhythmic Gestures**
- Groove 🎵 (head bobbing detection for music appreciation)

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

#### Rhythmic Gestures
- **Groove** 🎵
  - Gesture recognition for music appreciation
  - Features:
    1. Head movement tracking using nose landmarks
    2. Rhythmic pattern detection
    3. Directional change analysis
    4. Adaptive threshold system
  - Implementation details:
    - Weighted movement calculation (1.5x horizontal, 0.8x vertical)
    - 8-frame movement buffer for quick detection
    - 2-second gesture persistence
    - Compatible with other gesture detections

### Usage
The application will launch in fullscreen mode. Use the following controls:
- `q` - Quit the application
- `r` - Reset tracking calibration (eyebrows and head movement)
- `esc` - Exit fullscreen

#### On-Screen Information
- Real-time gesture recognition status
- Available gesture list
- Visual feedback for detected features
- Gesture persistence indicators

#### Roadmap
- [ ] Additional multimodal gestures
- [ ] Gesture combination sequences
- [ ] Custom gesture training interface
- [ ] Performance optimizations
- [ ] Configuration UI
- [ ] Advanced gesture analytics
- [ ] Enhanced rhythm detection capabilities
- [ ] Multi-gesture visualization system

### Technical Details
#### Multimodal Integration
- Real-time fusion of multiple input streams
- Adaptive threshold management
- Temporal smoothing for stable detection
- State management for complex gestures
- Priority-based gesture handling

#### Performance Features
- Efficient landmark processing
- Optimized detection pipelines
- Smart gesture state management
- Weighted movement analysis
- Gesture cooldown system

### Key Notes
This project showcases computer vision and gesture recognition techniques. The gestures were chosen for their detection reliability and technical suitability, without intent to define or standardize their meanings, acknowledging cultural variations.

#### Intended Use
- Research and academic purposes
- Technical demonstrations
- Computer vision development

### Testing
The project includes a comprehensive test suite using pytest. Tests cover gesture recognition accuracy, system robustness, and performance metrics.

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest-v
```

For detailed test coverage: `pytest --cov=gesture_recognition`

#### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file in the root directory of this repository for detailed terms and conditions.

---
*Built with OpenCV and MediaPipe*