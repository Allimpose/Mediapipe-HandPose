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
file = np.genfromtxt('/home/jun/Mediapipe-handpose/data/origin_data/onehand_gesture_train.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv2.ml.KNearest_create()
knn.train(angle, cv2.ml.ROW_SAMPLE, label)

def calculate_yaw(wrist, middle_mcp):
    # 손목 → 중지 MCP 벡터
    vector = np.array(middle_mcp) - np.array(wrist)

    # XY 평면에서의 Yaw 계산 (atan2 사용)
    yaw = np.arctan2(vector[1], vector[0])  # y/x

    # 각도를 도(degree)로 변환
    return np.degrees(yaw)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for res in results.multi_hand_landmarks:
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

            img_x = frame.shape[1]
            img_y = frame.shape[0]

            hand_x = res.landmark[0].x
            hand_y = res.landmark[0].y
            hand_z = res.landmark[0].z

            cv2.putText(frame, text = gesture[idx].upper(),
                       org = (int(hand_x * img_x), int(hand_y * img_y)+20),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2
                       )
            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

            # 손목(Wrist)와 중지 MCP(Middle Finger MCP) 좌표 추출
            landmarks = res.landmark
            wrist = [landmarks[mp_hands.HandLandmark.WRIST].x,
                     landmarks[mp_hands.HandLandmark.WRIST].y,
                     landmarks[mp_hands.HandLandmark.WRIST].z]
            middle_mcp = [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                          landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                          landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z]

            # Yaw (좌우 회전 각도) 계산
            yaw = calculate_yaw(wrist, middle_mcp)

            # 결과 출력
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Hand Yaw', frame)

    # 'q'를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
