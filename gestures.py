import cv2
import mediapipe as mp
import pyautogui

# --- Configuration ---
WEBCAM_ID = 0  # Change if you have multiple webcams
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

pyautogui.FAILSAFE = False

# --- Initialization ---
cap = cv2.VideoCapture(WEBCAM_ID)
if not cap.isOpened():
    print(f"Error: Could not open webcam with ID {WEBCAM_ID}")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,               
    min_detection_confidence=0.7,  
    min_tracking_confidence=0.6  
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

print("Starting Basic Hand Tracking...")
print(f"Screen Resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print("Press 'q' to quit.")

prev_index_tip_x = None
prev_index_tip_y = None

try:
    while True:
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image_rgb.shape

        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)

        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        # --- Hand Landmark Processing ---
        if results.multi_hand_landmarks:
            # Iterate through detected hands (though max_num_hands is 1 here)
            for hand_landmarks in results.multi_hand_landmarks:

                # --- 1. Draw Landmarks (Visual Feedback) ---
                mp_drawing.draw_landmarks(
                    image_bgr, # Draw on the BGR image for display
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # --- 2. Extract Landmark Data ---
                # Get specific landmarks (coordinates are normalized 0.0 -> 1.0)
                index_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip_landmark = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                
                
                if prev_index_tip_x is not None and prev_index_tip_y is not None:
                    # Calculate the new cursor position based on the index finger position
                    curr_index_tip_x = index_tip_landmark.x
                    curr_index_tip_y = index_tip_landmark.y
                    
                    # Calculate the distance moved
                    dx = (curr_index_tip_x - prev_index_tip_x)
                    dy = (curr_index_tip_y - prev_index_tip_y)
                    if abs(dx) < 0.5 or abs(dy) < 0.5:
                        pyautogui.moveRel(dx * SCREEN_WIDTH, dy * SCREEN_HEIGHT)

                prev_index_tip_x = index_tip_landmark.x
                prev_index_tip_y = index_tip_landmark.y

        # --- Display the image ---
        cv2.imshow('Basic Hand Tracking Template', image_bgr)

        # --- Exit Condition ---
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed.")
            break

finally:
    # --- Release Resources ---
    print("Releasing resources...")
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    if 'hands' in locals() and hands:
        hands.close()
    print("Exited.")