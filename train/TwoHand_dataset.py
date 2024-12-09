import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

max_num_hands = 2

gesture = {
    0:'write', 1:'keyboard', 2:'smartphone'
} # 3 two hand Gestures

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3)


file = np.genfromtxt('/home/jun/Mediapipe-handpose/data/train_data/twohand_gesture_train.csv', delimiter=',')

cap = cv2.VideoCapture(0)
i = 0

file = np.array(["Langle1","Langle2","Langle3","Langle4","Langle5","Langle6","Langle7","Langle8","Langle9",
"Langle10","Langle11","Langle12","Langle13","Langle14","Langle15","Rangle1","Rangle2","Rangle3","Rangle4",
"Rangle5","Rangle6","Rangle7","Rangle8","Rangle9","Rangle10","Rangle11","Rangle12","Rangle13","Rangle14","Rangle15","Class"])

def Lclick(event, x, y, flags, param):
    global Alldata, file
    global i
    if event == cv2.EVENT_MOUSEWHEEL :
        file = np.vstack((file, Alldata))
        print(file.shape)
    elif event == cv2.EVENT_MBUTTONDOWN:
        i += 1
        print("gesture number: ",i)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', Lclick)

while cap.isOpened() :

    angleR = [0 for i in range(15)]
    angleL = [0 for i in range(15)]
    ret, img = cap.read()
    img = cv2.flip(img,1)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # hand detect
    result = hands.process(img)
    hand_landmarks = result.multi_hand_landmarks
    handedness = result.multi_handedness

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if not ret :
        break

    if result.multi_hand_landmarks is not None :

        for idx, hand_handedness in enumerate(result.multi_handedness):
            if len(handedness) == 2:
                if hand_handedness.classification[0].label == "Left":
                    LWhich = 1
                    for res in result.multi_hand_landmarks :
                        joint = np.zeros((21, 3))
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    for j, lm in enumerate(result.multi_hand_landmarks[0].landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    v1L = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2L = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    vL = v2L - v1L # [20,3]
                    # Normalize v
                    vL = vL / np.linalg.norm(vL, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angleL = np.arccos(np.einsum('nt,nt->n',
                        vL[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        vL[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]
                    angleL = np.degrees(angleL)

                if hand_handedness.classification[0].label == "Right":
                    RWhich = 2
                    for res in result.multi_hand_landmarks :

                        joint2 = np.zeros((21, 3))

                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    for j2, lm2 in enumerate(result.multi_hand_landmarks[1].landmark):
                        joint2[j2] = [lm2.x, lm2.y, lm2.z]

                    v1R = joint2[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2R = joint2[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    vR = v2R - v1R # [20,3]

                    vR = vR / np.linalg.norm(vR, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product

                    angleR = np.arccos(np.einsum('nt,nt->n',
                        vR[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        vR[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angleR = np.degrees(angleR)


                anglesum_LR = np.concatenate((angleL,angleR))
                anglesum_RL = np.concatenate((angleR,angleL))

                data_LR = np.array([anglesum_LR], dtype=np.float32)
                data_RL = np.array([anglesum_RL], dtype=np.float32)

                data_LR = np.append(data_LR, i)
                data_RL = np.append(data_RL, i)

                Alldata = np.vstack((data_LR, data_RL))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Dataset', img)

cap.release()
cv2.destroyAllWindows()
np.savetxt('/home/jun/Mediapipe-handpose/data/train_data/twohand_gesture_train.csv', file, fmt='%s', delimiter=',')
