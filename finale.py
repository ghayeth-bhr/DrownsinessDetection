import threading
import time
import winsound
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp


class DrowsinessDetector:
    def __init__(self):
        self.yawn_state = ''
        self.left_eye_state = ''
        self.right_eye_state = ''
        self.alert_text = ''

        self.blinks = 0
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_duration = 0

        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False

        # Initialize models
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False  # Better for video
        )
        self.points_ids = [187, 411, 152, 68, 174, 399, 298]

        self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
        self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        time.sleep(1.0)

        # Frame sharing between threads
        self.latest_raw_frame = None
        self.latest_processed_frame = None
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()

        # Create threads
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.display_thread = threading.Thread(target=self.display_frames)

        # Start threads
        self.capture_thread.start()
        self.process_thread.start()
        self.display_thread.start()

    def update_info(self, frame):
        # Create a black background for the info panel
        info_panel = np.zeros((480, 300, 3), dtype=np.uint8)

        # Set alert text and color
        alert_color = (0, 0, 255)  # Default red
        if round(self.yawn_duration, 2) > 0.30:
            self.alert_text = "ALERT: Prolonged Yawn Detected!"
            alert_color = (0, 165, 255)  # Orange
            self.play_sound_in_thread()

        if round(self.microsleeps, 2) > 0.50:
            self.alert_text = "ALERT: Microsleep Detected!"
            alert_color = (0, 0, 255)  # Red
            self.play_sound_in_thread()

        # Add text to the info panel
        y_offset = 30
        line_height = 30

        # Title
        cv2.putText(info_panel, "Drowsiness Detector", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += line_height * 2

        # Alert if present
        if self.alert_text:
            cv2.putText(info_panel, self.alert_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
            y_offset += line_height * 2

        # Stats
        stats = [
            f"Blinks: {self.blinks}",
            f"Microsleeps: {round(self.microsleeps, 2)}s",
            f"Yawns: {self.yawns}",
            f"Yawn Duration: {round(self.yawn_duration, 2)}s"
        ]

        for stat in stats:
            cv2.putText(info_panel, stat, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += line_height

        # Combine the video frame and info panel
        combined = np.hstack((frame, info_panel))
        return combined

    def predict_eye(self, eye_frame, eye_state):
        # Resize for faster processing
        eye_frame = cv2.resize(eye_frame, (64, 64))
        results_eye = self.detecteye.predict(eye_frame, verbose=False)
        boxes = results_eye[0].boxes
        if len(boxes) == 0:
            return eye_state

        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        if class_id == 1:
            eye_state = "Close Eye"
        elif class_id == 0 and confidences[max_confidence_index] > 0.30:
            eye_state = "Open Eye"

        return eye_state

    def predict_yawn(self, yawn_frame):
        # Resize for faster processing
        yawn_frame = cv2.resize(yawn_frame, (128, 128))
        results_yawn = self.detectyawn.predict(yawn_frame, verbose=False)
        boxes = results_yawn[0].boxes

        if len(boxes) == 0:
            return self.yawn_state

        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy()
        max_confidence_index = np.argmax(confidences)
        class_id = int(class_ids[max_confidence_index])

        if class_id == 0:
            self.yawn_state = "Yawn"
        elif class_id == 1 and confidences[max_confidence_index] > 0.50:
            self.yawn_state = "No Yawn"

    def capture_frames(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
            else:
                break

    def process_frames(self):
        while not self.stop_event.is_set():
            with self.frame_lock:
                if self.latest_raw_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.latest_raw_frame.copy()

            # Process the frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    ih, iw, _ = frame.shape
                    points = []

                    for point_id in self.points_ids:
                        lm = face_landmarks.landmark[point_id]
                        x, y = int(lm.x * iw), int(lm.y * ih)
                        points.append((x, y))

                    if len(points) != 0:
                        x1, y1 = points[0]
                        x2, _ = points[1]
                        _, y3 = points[2]

                        x4, y4 = points[3]
                        x5, y5 = points[4]

                        x6, y6 = points[5]
                        x7, y7 = points[6]

                        x6, x7 = min(x6, x7), max(x6, x7)
                        y6, y7 = min(y6, y7), max(y6, y7)

                        mouth_roi = frame[y1:y3, x1:x2]
                        right_eye_roi = frame[y4:y5, x4:x5]
                        left_eye_roi = frame[y6:y7, x6:x7]

                        try:
                            self.left_eye_state = self.predict_eye(left_eye_roi, self.left_eye_state)
                            self.right_eye_state = self.predict_eye(right_eye_roi, self.right_eye_state)
                            self.predict_yawn(mouth_roi)

                        except Exception as e:
                            print(f"Error in prediction: {e}")

                        if self.left_eye_state == "Close Eye" and self.right_eye_state == "Close Eye":
                            if not self.left_eye_still_closed and not self.right_eye_still_closed:
                                self.left_eye_still_closed, self.right_eye_still_closed = True, True
                                self.blinks += 1
                            self.microsleeps += 1 / 30  # Assuming ~30 FPS processing
                        else:
                            if self.left_eye_still_closed and self.right_eye_still_closed:
                                self.left_eye_still_closed, self.right_eye_still_closed = False, False
                            self.microsleeps = 0

                        if self.yawn_state == "Yawn":
                            if not self.yawn_in_progress:
                                self.yawn_in_progress = True
                                self.yawns += 1
                            self.yawn_duration += 1 / 30  # Assuming ~30 FPS processing
                        else:
                            if self.yawn_in_progress:
                                self.yawn_in_progress = False
                                self.yawn_duration = 0

            # Create processed frame with info panel
            processed_frame = self.update_info(frame)
            with self.frame_lock:
                self.latest_processed_frame = processed_frame

    def display_frames(self):
        while not self.stop_event.is_set():
            with self.frame_lock:
                if self.latest_raw_frame is None:
                    time.sleep(0.01)
                    continue
                raw_frame = self.latest_raw_frame.copy()
                processed_frame = self.latest_processed_frame

            # Show raw camera feed (fast)
            cv2.imshow('Live Camera Feed', raw_frame)

            # Show processed feed (may be slower)
            if processed_frame is not None:
                cv2.imshow('Drowsiness Detection', processed_frame)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:  # 'q' or ESC
                self.stop_event.set()
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def play_alert_sound(self):
        frequency = 1000
        duration = 500
        winsound.Beep(frequency, duration)

    def play_sound_in_thread(self):
        sound_thread = threading.Thread(target=self.play_alert_sound)
        sound_thread.start()


if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.capture_thread.join()
    detector.process_thread.join()
    detector.display_thread.join()