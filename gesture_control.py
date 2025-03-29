import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
from enum import Enum

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

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Presentation control variables
presentation_state = PresentationState.NOT_STARTED
last_gesture_time = 0
gesture_delay = 1.0  # Delay between gestures in seconds
current_gesture = GestureType.NONE
last_detected_gesture = GestureType.NONE

# Gesture thresholds (optimized for PowerPoint)
PINCH_THRESHOLD = 0.05
SWIPE_THRESHOLD_X = 0.2
SWIPE_THRESHOLD_Y = 0.1
FIST_FINGER_CURL_THRESHOLD = 0.1
OPEN_HAND_THRESHOLD = 0.15

def get_current_gesture(hand_landmarks):
    landmarks = hand_landmarks.landmark
    wrist = landmarks[mp_hands.HandLandmark.WRIST]
    
    # Get key points
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # 1. Check for pinch (highest priority)
    pinch_distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    if pinch_distance < PINCH_THRESHOLD:
        return GestureType.PINCH
    
    # 2. Check for swipes (both left and right)
    # Calculate finger positions relative to wrist
    index_rel_x = index_tip.x - wrist.x
    middle_rel_x = middle_tip.x - wrist.x
    
    # Swipe left (fingers to left of wrist)
    if (index_rel_x < -SWIPE_THRESHOLD_X and 
        middle_rel_x < -SWIPE_THRESHOLD_X and
        abs(index_tip.y - middle_tip.y) < SWIPE_THRESHOLD_Y):
        return GestureType.SWIPE_LEFT
    
    # Swipe right (fingers to right of wrist)
    if (index_rel_x > SWIPE_THRESHOLD_X and 
        middle_rel_x > SWIPE_THRESHOLD_X and
        abs(index_tip.y - middle_tip.y) < SWIPE_THRESHOLD_Y):
        return GestureType.SWIPE_RIGHT
    
    # 3. Check for fist
    is_fist = True
    tips = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
            mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
            mp_hands.HandLandmark.PINKY_TIP]
    mcp_joints = [mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.INDEX_FINGER_MCP,
                 mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_MCP,
                 mp_hands.HandLandmark.PINKY_MCP]
    
    for tip, mcp in zip(tips, mcp_joints):
        if landmarks[tip].y < landmarks[mcp].y - FIST_FINGER_CURL_THRESHOLD:
            is_fist = False
            break
    
    if is_fist:
        return GestureType.FIST
    
    # 4. Check for open hand
    is_open = True
    for tip, pip in zip(tips, [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
                              mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_PIP,
                              mp_hands.HandLandmark.PINKY_PIP]):
        if landmarks[tip].y > landmarks[pip].y + OPEN_HAND_THRESHOLD:
            is_open = False
            break
    
    if is_open:
        return GestureType.OPEN_HAND
    
    return GestureType.NONE

def handle_gestures():
    global presentation_state, last_gesture_time, current_gesture, last_detected_gesture
    
    current_time = time.time()
    
    # Only process new gestures after the delay period
    if current_time - last_gesture_time < gesture_delay:
        return
    
    # Only act if we have a new gesture that's different from last one
    if current_gesture != last_detected_gesture and current_gesture != GestureType.NONE:
        last_detected_gesture = current_gesture
        last_gesture_time = current_time
        
        if current_gesture == GestureType.PINCH:
            if presentation_state == PresentationState.NOT_STARTED:
                # Start PowerPoint presentation
                pyautogui.hotkey('command', 'enter')
                time.sleep(1)  # Wait for presentation to start
                presentation_state = PresentationState.RUNNING
                print("Presentation started")
            elif presentation_state == PresentationState.RUNNING:
                pyautogui.press('right')  # Next slide
                print("Next slide")
                
        elif current_gesture == GestureType.SWIPE_LEFT:
            if presentation_state == PresentationState.RUNNING:
                pyautogui.press('left')  # Previous slide
                print("Previous slide")
                
        elif current_gesture == GestureType.SWIPE_RIGHT:
            if presentation_state == PresentationState.RUNNING:
                pyautogui.press('right')  # Next slide (alternative to pinch)
                print("Next slide (swipe)")
                
        elif current_gesture == GestureType.FIST:
            if presentation_state != PresentationState.NOT_STARTED:
                pyautogui.press('esc')  # Exit presentation
                presentation_state = PresentationState.NOT_STARTED
                print("Presentation exited")

def draw_ui(frame, state, gesture):
    """Draw user interface elements on the frame"""
    # Status indicator
    status_text = f"Status: {state.name.replace('_', ' ')}"
    cv2.putText(frame, status_text, (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Current gesture
    gesture_color = (0, 255, 0) if gesture != GestureType.NONE else (255, 255, 255)
    gesture_text = f"Gesture: {gesture.name.replace('_', ' ')}"
    cv2.putText(frame, gesture_text, (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    
    # Instructions
    instructions = [
        "Pinch: Start/Next Slide",
        "Swipe ←: Previous Slide",
        "Swipe →: Next Slide",
        "Fist: Exit Presentation"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (20, 110 + i*30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def main():
    global current_gesture
    
    print("Starting PowerPoint Gesture Control...")
    print("Make sure PowerPoint is in focus and ready.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            
            # Flip the frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks (fixed this line)
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                    )
                    
                    # Detect current gesture
                    current_gesture = get_current_gesture(hand_landmarks)
                    
                    # Handle gestures with debouncing
                    handle_gestures()
            
            # Draw UI
            draw_ui(frame, presentation_state, current_gesture)
            
            # Display the frame
            cv2.imshow('PowerPoint Gesture Control', frame)
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Gesture control system stopped.")

if __name__ == "__main__":
    main()