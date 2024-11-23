# Multimodal Gesture Recognition

A real-time computer vision solution for simultaneous recognition of hand gestures, facial expressions, and body postures using OpenCV and MediaPipe.

> 🚧 Active Development: This project is evolving. Core features are stable, but expect frequent enhancements.

## Overview

The project implements a comprehensive gesture recognition pipeline that processes multiple input modalities in real-time:
- Hand gesture tracking and classification
- Facial expression analysis
- Body posture detection

## Key Features

- Multi-threaded processing for optimal performance
- Configurable gesture detection thresholds
- Low-latency real-time feedback
- Modular architecture for easy extension

## Current Support

**Hand Gestures**
- Peace sign ✌️
- OK gesture 👌

**Facial Expressions**
- Smile detection 😊

**Body Postures**
- Shrug gesture 🤷

## Requirements

- Python 3.9+
- Webcam
- OpenCV compatible GPU (optional, improves performance)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/multimodal-gesture-recognition.git
cd multimodal-gesture-recognition

# Install dependencies
pip3 install -r requirements.txt

# Run the application
python3 gesture_recognition.py
```

## Usage

The application will launch in fullscreen mode. Use the following controls:
- `q` - Quit the application
- `esc` - Exit fullscreen

## Roadmap

- [ ] Additional gesture support
- [ ] Custom gesture training
- [ ] Performance optimizations
- [ ] Configuration UI
- [ ] Gesture sequence detection

## Contributing

Contributions are welcome! Check [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT

---
*Built with OpenCV and MediaPipe*