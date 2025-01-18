import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 2 # Number of recognizable hands
gesture = {
    0:'fist', 1:'point', 2:'index_click', 3:'middle_click', 4:'ring_click', 5:'pinky_click', 6:'hand_default'
} # 11 Gestures

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Gesture recognition data
file = np.genfromtxt('/home/jun/Mediapipe-HandPose/data/train_data/onehand_gui_train.csv', delimiter=',')

cap = cv2.VideoCapture(0)
i = 0

file = np.array(["angle1","angle2","angle3","angle4","angle5","angle6","angle7","angle8","angle9",
"angle10","angle11","angle12","angle13","angle14","angle15","Class"])

def Lclick(event, x, y, flags, param):
    global data, file
    global i
    if event == cv2.EVENT_MOUSEWHEEL:
        file = np.vstack((file, data))
        print(file.shape)
    elif event == cv2.EVENT_MBUTTONDOWN:
        i += 1
        print("gesture number: ",i)

cv2.namedWindow('Dataset')
cv2.setMouseCallback('Dataset', Lclick)


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks is not None:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark):
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

            angle = np.degrees(angle) # Convert radian to degree
            data = np.array([angle], dtype=np.float32)

            data = np.append(data, i) #
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Dataset', img)
    if cv2.waitKey(1) == ord('q'):
        break

np.savetxt('/home/jun/Mediapipe-HandPose/data/train_data/onehand_gui_train.csv', file, fmt='%s', delimiter=',')
