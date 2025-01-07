import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

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
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 손목(Wrist)와 중지 MCP(Middle Finger MCP) 좌표 추출
            landmarks = hand_landmarks.landmark
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
