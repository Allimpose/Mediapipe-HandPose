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

def calculate_yaw(wrist, middle_mcp):
    # 손목 → 중지 MCP 벡터
    vector = np.array(middle_mcp) - np.array(wrist)

    # XY 평면에서의 Yaw 계산 (atan2 사용)
    yaw = np.arctan2(vector[1], vector[0])  # y/x

    # 각도를 도(degree)로 변환
    return np.degrees(yaw)

def calculate_rotated_rectangle(circle_center, radius, offset, yaw, rect_width, rect_height):
    # 각도를 라디안으로 변환
    angle_rad = np.radians(yaw)

    # 사각형 중심 좌표 계산
    rect_center = (
        int(circle_center[0] + (radius + offset) * np.cos(angle_rad)),
        int(circle_center[1] + (radius + offset) * np.sin(angle_rad))
    )

    # 사각형 꼭짓점 좌표 정의 (중심 기준 상대 위치)
    rect_points = [
        (-rect_width // 2, -rect_height // 2),
        (rect_width // 2, -rect_height // 2),
        (rect_width // 2, rect_height // 2),
        (-rect_width // 2, rect_height // 2),
    ]

    # 회전 행렬 생성
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad), np.cos(angle_rad)]
    ])

    # 회전 행렬 적용 및 원점 기준 좌표 계산
    rotated_points = [np.dot(rotation_matrix, point) for point in rect_points]

    # 사각형 중심 좌표를 기준으로 최종 좌표 계산
    final_points = [(int(p[0] + rect_center[0]), int(p[1] + rect_center[1])) for p in rotated_points]

    return final_points

def is_fingers_inside_circle(landmarks, circle_center, radius):
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]

    for tip in finger_tips:
        tip_x = int(landmarks[tip].x * img_x)
        tip_y = int(landmarks[tip].y * img_y)
        distance = np.sqrt((circle_center[0] - tip_x) ** 2 + (circle_center[1] - tip_y) ** 2)
        if distance > radius:
            return False
    return True

def is_fingers_inside_rectangle(landmark, finger, rect_top_left, rect_bottom_right, img_width, img_height):
    finger_x = int(landmark[finger].x * img_width)
    finger_y = int(landmark[finger].y * img_height)

    # 직사각형 안에 있는지 확인
    if rect_top_left[0] <= finger_x <= rect_bottom_right[0] and rect_top_left[1] <= finger_y <= rect_bottom_right[1]:
        return True
    return False

final_points2 = None
yaw2 = None
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 가로
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) # 세로

current_zoom = cap.get(cv2.CAP_PROP_ZOOM)

cap.set(cv2.CAP_PROP_ZOOM, 1)

if not cap.get(cv2.CAP_PROP_ZOOM):
    print("Zoom control is not supported by this camera.")
else:
    print("Zoom control is supported.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for res in results.multi_hand_landmarks:
            joint = np.zeros((21, 3))  # 21 joints, create array to store xyz values

            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z]

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

            mp_drawing.draw_landmarks(frame, res, mp_hands.HAND_CONNECTIONS)

            img_x = frame.shape[1]
            img_y = frame.shape[0]

            hand_x = res.landmark[0].x
            hand_y = res.landmark[0].y
            hand_z = res.landmark[0].z

            landmarks = res.landmark
            wrist = [landmarks[mp_hands.HandLandmark.WRIST].x,
                     landmarks[mp_hands.HandLandmark.WRIST].y,
                     landmarks[mp_hands.HandLandmark.WRIST].z]
            middle_mcp = [landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
                          landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
                          landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z]

            yaw = calculate_yaw(wrist, middle_mcp)

            yaw3 = yaw + 90

            # yaw2의 초기 설정
            if yaw2 is None:
                yaw2 = -90

            # idx == 0일 때 한 번만 yaw2 업데이트
            if idx == 0 and not yaw2_updated and is_fingers_inside_circle(res.landmark, circle_center, radius):
                yaw2 += yaw3
                yaw2_updated = True  # 업데이트 플래그 설정
                final_points2 = calculate_rotated_rectangle(circle_center, radius, offset, yaw2, rect_width, rect_height)

            # idx가 0이 아닐 때 플래그를 리셋
            if idx != 0:
                yaw2_updated = False

            circle_center = (640, 360)
            radius = 120
            offset = 20
            rect_width = 50
            rect_height = 30

            if final_points2 is not None:
                cv2.polylines(frame, [np.array(final_points2)], isClosed=True, color=(255, 255, 0), thickness=3)

            final_points = calculate_rotated_rectangle(circle_center, radius, offset, yaw, rect_width, rect_height)

            pt1 = (10, 10)
            pt2 = (100, 100)

            pt3 = (1180, 10)
            pt4 = (1270, 100)

            width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

            if idx == 1 and is_fingers_inside_rectangle(res.landmark,mp_hands.HandLandmark.INDEX_FINGER_TIP,pt1, pt2,frame.shape[1], frame.shape[0]):
                current_zoom -= 1
                if current_zoom < 0:
                    current_zoom = 0
                cap.set(cv2.CAP_PROP_ZOOM, current_zoom)
                print(f"Zoom Out: {current_zoom}")
            elif idx == 1 and is_fingers_inside_rectangle(res.landmark,mp_hands.HandLandmark.INDEX_FINGER_TIP,pt3, pt4,frame.shape[1], frame.shape[0]):
                current_zoom += 1
                cap.set(cv2.CAP_PROP_ZOOM, current_zoom)
                print(f"Zoom In: {current_zoom}")

            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

            cv2.rectangle(frame, pt3, pt4, (0, 255, 0), 2)

            cv2.circle(frame, circle_center, radius, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.polylines(frame, [np.array(final_points)], isClosed=True, color=(0, 255, 0), thickness=3)
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, text = gesture[idx].upper(),
                       org = (int(hand_x * img_x), int(hand_y * img_y)+20),
                       fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2
                       )

    cv2.imshow('Hand Yaw', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()