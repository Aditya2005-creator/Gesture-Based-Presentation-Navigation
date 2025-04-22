import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import json
import logging
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GestureConfig:
    pinch_threshold: float = 0.08
    swipe_threshold_x: float = 0.15
    swipe_threshold_y: float = 0.15
    fist_finger_curl_threshold: float = 0.15
    open_hand_threshold: float = 0.2
    gesture_delay: float = 0.5
    smoothing_window: int = 3

class PresentationState(Enum):
    NOT_STARTED = 0
    RUNNING = 1
    PAUSED = 2

class GestureType(Enum):
    NONE = 0
    PINCH = 1
    SWIPE_LEFT = 2
    SWIPE_RIGHT = 3
    OPEN_HAND = 4
    FIST = 5
    POINT = 6  # New gesture for laser pointer

class GestureDetector:
    def __init__(self, config: GestureConfig):
        self.config = config
        self.gesture_history: List[GestureType] = []
        self.last_gesture_time = 0
        self.current_gesture = GestureType.NONE
        self.last_detected_gesture = GestureType.NONE
        self.laser_pointer_enabled = False
        self.screen_width, self.screen_height = pyautogui.size()
        
        # Initialize MediaPipe Hands
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.7,  # Lowered from 0.8
                min_tracking_confidence=0.5    # Lowered from 0.6
            )
            self.mp_drawing = mp.solutions.drawing_utils
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise

    def get_current_gesture(self, hand_landmarks) -> GestureType:
        try:
            landmarks = hand_landmarks.landmark
            
            # Get key points using proper indices
            wrist = landmarks[self.mp_hands.HandLandmark.WRIST]
            thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            # Debug information
            logger.debug(f"Thumb-Index distance: {np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)}")
            
            # 1. Check for FIST
            is_fist = True
            finger_tips = [
                self.mp_hands.HandLandmark.THUMB_TIP,
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                self.mp_hands.HandLandmark.RING_FINGER_TIP,
                self.mp_hands.HandLandmark.PINKY_TIP
            ]
            
            finger_pips = [
                self.mp_hands.HandLandmark.THUMB_IP,
                self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                self.mp_hands.HandLandmark.RING_FINGER_PIP,
                self.mp_hands.HandLandmark.PINKY_PIP
            ]
            
            for tip, pip in zip(finger_tips, finger_pips):
                tip_y = landmarks[tip].y
                pip_y = landmarks[pip].y
                if tip_y < pip_y - 0.02:  # Tip must be *below* pip (i.e., numerically greater)
                    is_fist = False
                    break
            
            # Additional check: all tips are close to the palm
            if is_fist:
                palm = landmarks[self.mp_hands.HandLandmark.WRIST]
                for tip in finger_tips:
                    dist = np.sqrt(
                        (landmarks[tip].x - palm.x) ** 2 +
                        (landmarks[tip].y - palm.y) ** 2
                    )
                    if dist > 0.25:  # Adjust threshold as needed
                        is_fist = False
                        break
            
            if is_fist:
                logger.debug("Detected FIST gesture")
                return GestureType.FIST
            
            # 2. Check for pinch (more lenient)
            pinch_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
            if pinch_distance < self.config.pinch_threshold:
                logger.debug("Detected PINCH gesture")
                return GestureType.PINCH
            
            # 3. Check for swipes (more lenient)
            index_rel_x = index_tip.x - wrist.x
            middle_rel_x = middle_tip.x - wrist.x
            
            # Check for horizontal movement with more lenient thresholds when laser is enabled
            swipe_threshold_y = self.config.swipe_threshold_y * (1.5 if self.laser_pointer_enabled else 1.0)
            swipe_threshold_x = self.config.swipe_threshold_x * (1.2 if self.laser_pointer_enabled else 1.0)
            
            if abs(index_tip.y - middle_tip.y) < swipe_threshold_y:
                if index_rel_x < -swipe_threshold_x and middle_rel_x < -swipe_threshold_x:
                    logger.debug("Detected SWIPE_LEFT gesture")
                    return GestureType.SWIPE_LEFT
                elif index_rel_x > swipe_threshold_x and middle_rel_x > swipe_threshold_x:
                    logger.debug("Detected SWIPE_RIGHT gesture")
                    return GestureType.SWIPE_RIGHT
            
            # 4. Check for point gesture
            is_pointing = True
            for finger in [self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                          self.mp_hands.HandLandmark.RING_FINGER_TIP,
                          self.mp_hands.HandLandmark.PINKY_TIP]:
                if landmarks[finger].y < landmarks[wrist].y - self.config.open_hand_threshold:
                    is_pointing = False
                    break
            
            if is_pointing and landmarks[index_tip].y < landmarks[wrist].y - self.config.open_hand_threshold:
                logger.debug("Detected POINT gesture")
                return GestureType.POINT
            
            return GestureType.NONE
            
        except Exception as e:
            logger.error(f"Error in gesture detection: {e}")
            return GestureType.NONE

    def smooth_gesture(self, gesture: GestureType) -> GestureType:
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.config.smoothing_window:
            self.gesture_history.pop(0)
        
        # Count occurrences of each gesture
        gesture_counts = {}
        for g in self.gesture_history:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        
        # Return most common gesture, but only if it appears more than once
        most_common = max(gesture_counts.items(), key=lambda x: x[1])
        if most_common[1] > 1:
            return most_common[0]
        return GestureType.NONE

    def handle_gestures(self, presentation_state: PresentationState) -> None:
        current_time = time.time()
        
        if current_time - self.last_gesture_time < self.config.gesture_delay:
            return
        
        if self.current_gesture != self.last_detected_gesture and self.current_gesture != GestureType.NONE:
            self.last_detected_gesture = self.current_gesture
            self.last_gesture_time = current_time
            
            try:
                if self.current_gesture == GestureType.PINCH:
                    # Toggle fullscreen
                    pyautogui.hotkey('command', 'enter')
                    logger.info("Toggled fullscreen")
                    
                elif self.current_gesture == GestureType.SWIPE_LEFT:
                    # Previous slide
                    pyautogui.press('left')
                    logger.info("Previous slide")
                    
                elif self.current_gesture == GestureType.SWIPE_RIGHT:
                    # Next slide
                    pyautogui.press('right')
                    logger.info("Next slide")
                    
                elif self.current_gesture == GestureType.FIST:
                    # Toggle laser pointer mode using Command+L
                    pyautogui.hotkey('command', 'l')
                    self.laser_pointer_enabled = not self.laser_pointer_enabled
                    logger.info(f"Laser pointer {'enabled' if self.laser_pointer_enabled else 'disabled'}")
                    
            except Exception as e:
                logger.error(f"Error executing gesture: {e}")

    def update_laser_pointer(self, hand_landmarks, frame):
        if not self.laser_pointer_enabled:
            return
            
        try:
            # Get index finger tip position
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            
            # Convert normalized coordinates to screen coordinates
            screen_x = int(index_tip.x * self.screen_width)
            screen_y = int(index_tip.y * self.screen_height)
            
            # Draw laser pointer on frame
            cv2.circle(frame, (int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])), 
                      10, (0, 0, 255), -1)  # Red circle
            
            # Move mouse cursor to pointer position
            pyautogui.moveTo(screen_x, screen_y)
            
        except Exception as e:
            logger.error(f"Error updating laser pointer: {e}")

def load_config(config_path: str = "gesture_config.json") -> GestureConfig:
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            return GestureConfig(**config_dict)
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using default settings.")
        return GestureConfig()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return GestureConfig()

def draw_ui(frame, state: PresentationState, gesture: GestureType, laser_enabled: bool):
    """Draw user interface elements on the frame"""
    # Status indicator with background
    status_text = f"Status: {state.name.replace('_', ' ')}"
    cv2.rectangle(frame, (10, 10), (300, 40), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Current gesture with background
    gesture_color = (0, 255, 0) if gesture != GestureType.NONE else (255, 255, 255)
    gesture_text = f"Gesture: {gesture.name.replace('_', ' ')}"
    cv2.rectangle(frame, (10, 50), (300, 80), (0, 0, 0), -1)
    cv2.putText(frame, gesture_text, (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    
    # Laser pointer status
    laser_color = (0, 255, 0) if laser_enabled else (0, 0, 255)
    laser_text = f"Laser: {'ON' if laser_enabled else 'OFF'}"
    cv2.rectangle(frame, (10, 90), (300, 120), (0, 0, 0), -1)
    cv2.putText(frame, laser_text, (20, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, laser_color, 2)
    
    # Instructions with background
    instructions = [
        "Pinch: Toggle Fullscreen",
        "Swipe ←: Previous Slide",
        "Swipe →: Next Slide",
        "Fist: Toggle Laser Pointer",
        "Press 'q' to quit"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.rectangle(frame, (10, 140 + i*30), (400, 170 + i*30), (0, 0, 0), -1)
        cv2.putText(frame, instruction, (20, 160 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    # Enable debug logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    config = load_config()
    detector = GestureDetector(config)
    presentation_state = PresentationState.NOT_STARTED
    
    # Initialize webcam with error handling
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Failed to open webcam")
    except Exception as e:
        logger.error(f"Failed to initialize webcam: {e}")
        return
    
    logger.info("Starting PowerPoint Gesture Control...")
    logger.info("Make sure PowerPoint is in focus and ready.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Flip the frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = detector.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks with different colors for better visibility
                    detector.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        detector.mp_hands.HAND_CONNECTIONS,
                        detector.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),  # Green for landmarks
                        detector.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)   # Blue for connections
                    )
                    
                    # Detect and handle gestures
                    detector.current_gesture = detector.get_current_gesture(hand_landmarks)
                    detector.handle_gestures(presentation_state)
                    
                    # Update laser pointer if enabled
                    detector.update_laser_pointer(hand_landmarks, frame)
            
            # Draw UI
            draw_ui(frame, presentation_state, detector.current_gesture, detector.laser_pointer_enabled)
            
            # Display the frame
            cv2.imshow('PowerPoint Gesture Control', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Gesture control system stopped.")

if __name__ == "__main__":
    main()
