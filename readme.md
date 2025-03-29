Gesture-Controlled PowerPoint Presentation
📝 Overview
A hands-free PowerPoint presentation control system using computer vision and hand gesture recognition. This application allows you to navigate through slides using intuitive hand gestures captured by your webcam.

✨ Features
Gesture-based navigation:

👌 Pinch to start presentation or go to next slide

👈 Swipe left for previous slide

👉 Swipe right for next slide (alternative to pinch)

✊ Fist to exit presentation

Real-time hand tracking visualization

Visual feedback showing detected gestures

Debounce system to prevent accidental rapid gestures

Cross-platform (Mac/Windows/Linux)

🛠️ Requirements
Python 3.7+

Webcam

Microsoft PowerPoint (or other presentation software)

Required Python packages:

Copy
opencv-python
mediapipe
pyautogui
numpy
🚀 Installation
Clone this repository or download the script

Install dependencies:

bash
Copy
pip install opencv-python mediapipe pyautogui numpy
🎮 Usage
Open your PowerPoint presentation

Run the script:

bash
Copy
python gesture_control.py
Perform gestures in view of your webcam:

Pinch (thumb and index finger together):

First pinch: Starts presentation (⌘+Enter)

Subsequent pinches: Next slide (→)

Swipe left (two fingers to left): Previous slide (←)

Swipe right (two fingers to right): Next slide (→)

Fist (closed hand): Exit presentation (Esc)

Press 'q' to quit the application

⚙️ Configuration
You can adjust these parameters in the code for better performance:

gesture_delay: Time between gesture actions (seconds)

PINCH_THRESHOLD: Sensitivity for pinch detection

SWIPE_THRESHOLD_X/Y: Sensitivity for swipe detection

🖥️ System Architecture
mermaid
Copy
graph TD
    A[Webcam Input] --> B[Hand Detection]
    B --> C[Gesture Recognition]
    C --> D[PowerPoint Control]
    D --> E[Visual Feedback]
📊 Gesture Detection Logic
Pinch Detection: Measures distance between thumb and index finger tips

Swipe Detection: Tracks finger positions relative to wrist

Fist Detection: Checks finger curl positions

Priority System: Pinch > Swipe > Fist

🛑 Known Limitations
Requires good lighting conditions

Works best with one hand in frame

May need calibration for different hand sizes

PowerPoint must be the active window

🤝 Contributing
Contributions are welcome! Please open an issue or pull request for any:

Bug fixes

New gesture implementations

Performance improvements

Additional features

📜 License
MIT License - Free for personal and educational use

💡 Tip: For best results, position your hand clearly in the webcam view and make deliberate gestures. The system includes a 1-second cooldown between gestures to prevent accidental activations.

