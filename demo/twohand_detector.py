import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

max_num_hands = 2

gesture = {
    0:'write', 1:'keyboard', 2:'smartphone'
} # 3 two hand Gestures

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)


file = np.genfromtxt('/home/jun/Mediapipe-handpose/data/origin_data/twohand_gesture_train.csv', delimiter=',')
anglesum = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(anglesum, cv2.ml.ROW_SAMPLE, label)

cap = cv2.VideoCapture(0)

while cap.isOpened() :
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
        print("------------------------------------------")
        if len(handedness) == 2:
            for res in result.multi_hand_landmarks :
                joint = np.zeros((21, 3))
                joint2 = np.zeros((21, 3))
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            for j, lm in enumerate(result.multi_hand_landmarks[0].landmark):
                joint[j] = [lm.x, lm.y, lm.z]

            for j2, lm2 in enumerate(result.multi_hand_landmarks[1].landmark):
                joint2[j2] = [lm2.x, lm2.y, lm2.z]

            v1L = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2L = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            vL = v2L - v1L # [20,3]
            # Normalize v
            vL = vL / np.linalg.norm(vL, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angleL = np.arccos(np.einsum('nt,nt->n',
                vL[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                vL[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            v1R = joint2[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent joint
            v2R = joint2[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            vR = v2R - v1R # [20,3]
            # Normalize v
            vR = vR / np.linalg.norm(vR, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angleR = np.arccos(np.einsum('nt,nt->n',
                vR[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                vR[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

            anglesum = np.concatenate((angleL,angleR))

            anglesum = np.degrees(anglesum)
            data = np.array([anglesum], dtype=np.float32)

            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            # 인식된 제스쳐 표현하기
            img_x = img.shape[1]
            img_y = img.shape[0]

            hand_x = res.landmark[0].x
            hand_y = res.landmark[0].y
            hand_z = res.landmark[0].z

            cv2.putText(img, text = gesture[idx].upper(),
                       org = (int(hand_x * img_x), int(hand_y * img_y)+20),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2
                       )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('Dataset', img)

cap.release()
cv2.destroyAllWindows()
