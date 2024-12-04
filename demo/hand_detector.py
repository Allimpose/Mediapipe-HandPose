import cv2
import mediapipe as mp
import numpy as np

max_num_hands = 2
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'gun', 8:'spiderman', 9:'yeah', 10:'ok'
}

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8)

# Gesture recognition model
file = np.genfromtxt('/home/jun/Mediapipe-HandPose/data/origin_data/onehand_gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

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

    # mark hand
    if result.multi_hand_landmarks is not None :

        # 이미지에 손 표현하기
        for res in result.multi_hand_landmarks :
            joint = np.zeros((21, 3)) # 21 joints, create array to store xyz values

            for j, lm in enumerate(res.landmark) :
                joint[j] = [lm.x, lm.y, lm.z]
            # Get joint number to connect
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v = v2 - v1 # Bone values (x, y, z coordinate values → vector values)

            # Value of bone (straight value)
            v = v / np.linalg.norm(v, axis = 1)[:, np.newaxis]
            # Calculate the angle between bones by the value of bones, 15 with large changes
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

            angle = np.degrees(angle)

            data = np.array([angle], dtype=np.float32)
            ret, results, neighbours, dist = knn.findNearest(data, 3)
            idx = int(results[0][0])

            img_x = img.shape[1]
            img_y = img.shape[0]
            
            hand_x = res.landmark[0].x
            hand_y = res.landmark[0].y
            hand_z = res.landmark[0].z

            cv2.putText(img, text = gesture[idx].upper(),
                       org = (int(hand_x * img_x), int(hand_y * img_y)+20),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2
                       ) 
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

    if cv2.waitKey(1) == ord('q'):
        break
    cv2.imshow('hand', img)

cap.release()
cv2.destroyAllWindows()
