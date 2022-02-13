import cv2
import mediapipe as mp
import numpy as np
import os
from pprint import pprint

def euclidean_distance(a, b):
    return np.linalg.norm(a-b)

def main():

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_hand_landmarks_styles = mp_draw.DrawingSpec(color=(0, 0, 255), thickness=5)
    mp_hand_connectionss_styles = mp_draw.DrawingSpec(color=(0, 255, 0), thickness=10)
    hands = mp_hands.Hands()

    gesture_names = []
    gestures = []
    for line in open(os.path.join("pose", "gestures.txt"), "r").readlines():
        line = line.strip()

        name, *gesture = line.split(",")
        gesture_names.append(name.strip())

        gesture = np.array(list(map(float, gesture)))
        gesture = gesture / np.max(gesture)
        gestures.append(gesture)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width = frame.shape[:2]

        blob = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(blob)

        message = "No Hands Detected!"

        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) > 1:
                message = "Too Many Hands Detected!"
            else:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_hand_landmarks_styles,
                        mp_hand_connectionss_styles)

                    gesture = np.array([[landmark.x, landmark.y] for landmark in hand_landmarks.landmark])
                    # Compute gesture signature
                    signature = np.array([euclidean_distance(landmark, gesture[0]) for landmark in gesture])
                    signature = signature / np.max(signature)
                    # Signature distances from each of the saved gestrues
                    distances = [euclidean_distance(signature, gesture) for gesture in gestures]
                    message = f"Hands Detected: {gesture_names[np.argmin(distances)]}"

                    # for i, landmark in enumerate(hand_landmarks.landmark):
                    #     gesture.append([landmark.x, landmark.y])
                    #     x = int(landmark.x * frame_width)
                    #     y = int(landmark.y * frame_height)
                    #     cv2.putText(frame, f"{i}", (x - 25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.putText(frame, message, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

        cv2.imshow("video", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
