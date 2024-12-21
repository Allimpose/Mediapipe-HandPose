import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

max_num_hands = 2

gesture = {
    0:'write', 1:'keyboard', 2:'smartphone', 3:'point', 4:'V'
} # 5 two hand Gestures

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)


file = np.genfromtxt('/home/jun/Mediapipe-handpose/data/origin_data/twohandonehand_gesture_train.csv', delimiter=',')
anglesum_LR = file[:,:-1].astype(np.float32)
anglesum_RL = file[:,:-1].astype(np.float32)
angleA = file[:,:-1].astype(np.float32)
angleB = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(anglesum_LR, cv2.ml.ROW_SAMPLE, label)
knn.train(anglesum_RL, cv2.ml.ROW_SAMPLE, label)

knn.train(angleB, cv2.ml.ROW_SAMPLE, label)
knn.train(angleA, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened() :
    angleR = [0 for i in range(15)]
    angleL = [0 for i in range(15)]
    angleA = [0 for i in range(30)]
    angleB = [0 for i in range(30)]
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
        for res in result.multi_hand_landmarks :
            for idx, hand_handedness in enumerate(result.multi_handedness):
                if len(handedness) == 1:
                    joint = np.zeros((21, 3))
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                    for j, lm in enumerate(result.multi_hand_landmarks[0].landmark):
                        joint[j] = [lm.x, lm.y, lm.z]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                    #print("v1:",v1)
                    v = v2 - v1 # [20,3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angleA[0:15]=angle
                    angleB[15:30]=angle

                    angleA = np.degrees(angleA) # Convert radian to degree
                    angleB = np.degrees(angleB) # Convert radian to degree

                    dataA = np.array([angleA], dtype=np.float32)
                    dataB = np.array([angleB], dtype=np.float32)

                    ret_A, results_A, neighbours_A, dist_A = knn.findNearest(dataA, 3)
                    ret_B, results_B, neighbours_B, dist_B = knn.findNearest(dataB, 3)

                    idx_A = int(results_A[0][0])
                    idx_B = int(results_B[0][0])


                    img_x = img.shape[1]
                    img_y = img.shape[0]

                    hand_x = res.landmark[0].x
                    hand_y = res.landmark[0].y
                    hand_z = res.landmark[0].z

                    if idx_A == idx_B :
                        cv2.putText(img, text = gesture[idx_A].upper(),
                                   org = (int(hand_x * img_x), int(hand_y * img_y)-20),
                                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2
                                   )
                        cv2.putText(img, text = gesture[idx_B].upper(),
                                   org = (int(hand_x * img_x), int(hand_y * img_y)-50),
                                   fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2
                                   )
                    if idx_A == 3 or idx_B == 3 :
                        point_x = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                        point_y = result.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

                        cv2.circle(img, (int(point_x * img_x), int(point_y * img_y)), 10, (255,0,0), 2)

                if len(handedness) == 2:
                    if hand_handedness.classification[0].label == "Left":
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
                        joint2 = np.zeros((21, 3))
                        mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                        for j2, lm2 in enumerate(result.multi_hand_landmarks[1].landmark):
                            joint2[j2] = [lm2.x, lm2.y, lm2.z]

                        v1R = joint2[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
                        v2R = joint2[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
                        vR = v2R - v1R # [20,3]
                        # Normalize v
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

                        ret_LR, results_LR, neighbours_LR, dist_LR = knn.findNearest(data_LR, 3)
                        ret_RL, results_RL, neighbours_RL, dist_RL = knn.findNearest(data_RL, 3)

                        idx_LR = int(results_LR[0][0])
                        idx_RL = int(results_RL[0][0])

                        img_x = img.shape[1]
                        img_y = img.shape[0]

                        hand_x = res.landmark[0].x
                        hand_y = res.landmark[0].y
                        hand_z = res.landmark[0].z

                        if idx_LR == idx_RL :
                            cv2.putText(img, text = gesture[idx_LR].upper(),
                                       org = (int(hand_x * img_x), int(hand_y * img_y)+20),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2
                                       )
                            cv2.putText(img, text = gesture[idx_RL].upper(),
                                       org = (int(hand_x * img_x), int(hand_y * img_y)+50),
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2
                                       )



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Dataset', img)

cap.release()
cv2.destroyAllWindows()
