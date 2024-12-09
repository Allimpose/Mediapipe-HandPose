import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

max_num_hands = 2

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
while cap.isOpened() :
    ret, img = cap.read()
    img = cv2.flip(img,1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # hand detect
    result = hands.process(img)
    hand_landmarks = result.multi_hand_landmarks
    handedness = result.multi_handedness

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ax.clear()
    if not ret :
        break
    x1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    y1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    z1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    x2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    y2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    z2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

    if result.multi_hand_landmarks is not None :


        for res in result.multi_hand_landmarks :
            landmarks = res.landmark
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            x1 = [landmark.x for landmark in landmarks]
            y1 = [landmark.y for landmark in landmarks]
            z1 = [landmark.z for landmark in landmarks]

            x=[x1, x1]
            y=[y1, y1]
            z=[z1, z1]
            ax.scatter(x,y,z)

        # Set plot limits and labels
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')

        # Show plot
        plt.pause(0.01)
        plt.show(block=False)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('hand', img)

cap.release()
cv2.destroyAllWindows()
