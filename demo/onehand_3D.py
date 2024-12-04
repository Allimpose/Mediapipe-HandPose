import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PoseEstimator:
    def __init__(self):
        self.mp_pose = mp.solutions.hands
        self.pose = self.mp_pose.Hands()
        self.mp_drawing = mp.solutions.drawing_utils

    def process_webcam(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        cap = cv2.VideoCapture(0)  # Use default webcam (index 0)
        while cap.isOpened():
            ret, frame = cap.read()
            #frame = cv2.flip(frame,1)
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.results = self.pose.process(frame_rgb)
            annotated_frame = self._draw_landmarks(frame, self.results.multi_hand_landmarks)
            cv2.imshow('Pose Estimation', annotated_frame)


            ax.clear()

            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    landmarks = handLms.landmark

                x = [landmark.x for landmark in landmarks]
                y = [landmark.y for landmark in landmarks]
                z = [landmark.z for landmark in landmarks]
                # Plot 3D landmarks
                ax.scatter(x, y, z)

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
        cap.release()
        cv2.destroyAllWindows()
        
    def _draw_landmarks(self, image, landmarks):
        if landmarks is not None:
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    annotated_image = image.copy()
                    landmarks = handLms.landmark
                    self.mp_drawing.draw_landmarks(annotated_image, handLms, self.mp_pose.HAND_CONNECTIONS)
            return annotated_image
        else:
            return image

pose = PoseEstimator()
pose.process_webcam()
