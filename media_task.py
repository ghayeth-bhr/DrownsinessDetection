# -*- coding: utf-8 -*-
"""
Drowsiness Detection using Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR)
with MediaPipe Face Landmarker Task.

Detects blinks, microsleeps (prolonged eye closure), and yawns (prolonged mouth opening).
Provides visual feedback and optional sound alerts.
"""

import threading
import time
import cv2
import numpy as np
import mediapipe as mp
import winsound

try:
    SOUND_AVAILABLE = True
except ImportError:
    print("Warning: winsound module not found. Alert sounds will be disabled.")
    SOUND_AVAILABLE = False


class DrowsinessDetector:
    def __init__(self, task_model_path='face_landmarker.task'):
        # --- Configuration ---
        self.EAR_THRESHOLD = 0.15  # Eye Aspect Ratio threshold for blink/closure detection
        self.MAR_THRESHOLD = 0.35  # Mouth Aspect Ratio threshold for yawn detection
        self.MIN_YAWN_FRAMES = 10  # Consecutive frames MAR must be above threshold
        self.MICROSLEEP_ALERT_THRESHOLD = 0.7  # Duration (seconds) for microsleep alert
        self.YAWN_ALERT_THRESHOLD = 2.0  # Duration (seconds) for prolonged yawn alert

        # --- State Variables ---
        self.alert_text = ""
        self.blinks = 0
        self.yawns = 0
        self.microsleep_start_time = None
        self.microsleep_duration = 0.0
        self.yawn_duration = 0.0
        self.yawn_frames = 0
        self.yawn_start_time = None
        self.left_eye_still_closed = False
        self.right_eye_still_closed = False
        self.yawn_in_progress = False
        self.alert_active = False

        # --- MediaPipe Setup ---
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1
        )

        self.landmarker = FaceLandmarker.create_from_options(options)
        print("MediaPipe FaceLandmarker loaded successfully.")

        # Landmark indices (MediaPipe uses different indices than some TFLite models)
        self.LEFT_EAR_POINTS = [33, 160, 158, 133, 153, 144]  # Left eye points
        self.RIGHT_EAR_POINTS = [362, 385, 387, 263, 373, 380]  # Right eye points
        self.MOUTH_INDICES = [61, 291, 13, 14]  # Mouth points

        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.stop_event = threading.Event()
            self.stop_event.set()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Webcam opened. Reported FPS: {self.actual_fps if self.actual_fps > 0 else 'N/A'}")
        time.sleep(1.0)

        # --- Threading ---
        self.latest_raw_frame = None
        self.latest_processed_frame = None
        self.frame_lock = threading.Lock()
        self.stop_event = threading.Event()

        # --- Start Threads ---
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        self.display_thread = threading.Thread(target=self.display_frames)

        self.capture_thread.start()
        self.process_thread.start()
        self.display_thread.start()

    def process_frame_with_mediapipe(self, frame):
        """Process frame using MediaPipe and return landmarks."""
        # Convert the frame to MediaPipe's Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        # Process the image
        result = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if len(result.face_landmarks) == 0:
            return None

        # Get the first face's landmarks
        landmarks = result.face_landmarks[0]

        # Convert landmarks to pixel coordinates
        h, w = frame.shape[:2]
        landmark_points = []
        for landmark in landmarks:
            x = landmark.x * w
            y = landmark.y * h
            landmark_points.append((x, y))

        return np.array(landmark_points)

    def calculate_ear(self, eye_landmarks):
        """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
        if eye_landmarks is None or len(eye_landmarks) != 6:
            return 1.0

        if not np.all(np.isfinite(eye_landmarks)):
            return 1.0

        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

        if C < 1e-6:
            return 1.0

        ear = (A + B) / (2.0 * C)
        return ear

    def calculate_mar(self, mouth_landmarks):
        """Calculates the Mouth Aspect Ratio (MAR)."""
        if mouth_landmarks is None or len(mouth_landmarks) < 4:
            return 0.0

        if not np.all(np.isfinite(mouth_landmarks)):
            return 0.0

        left_corner = mouth_landmarks[0]
        right_corner = mouth_landmarks[1]
        top_lip_inner = mouth_landmarks[2]
        bottom_lip_inner = mouth_landmarks[3]

        horizontal_dist = np.linalg.norm(left_corner - right_corner)
        vertical_dist = np.linalg.norm(top_lip_inner - bottom_lip_inner)

        if horizontal_dist < 1e-6:
            return 0.0

        mar = vertical_dist / horizontal_dist
        return mar

    def update_info(self, frame):
        """Adds statistical information and alerts to the frame."""
        if frame is None:
            return np.zeros((480, 300, 3), dtype=np.uint8)

        height, _, _ = frame.shape
        info_panel_width = 300
        info_panel = np.zeros((height, info_panel_width, 3), dtype=np.uint8)

        # --- Alert Logic ---
        self.alert_text = ""
        current_alert = False
        alert_color = (0, 255, 255)

        if self.microsleep_duration > self.MICROSLEEP_ALERT_THRESHOLD:
            self.alert_text = f"ALERT: Microsleep ({self.microsleep_duration:.1f}s)"
            alert_color = (0, 0, 255)
            current_alert = True
        elif self.yawn_in_progress and self.yawn_duration > self.YAWN_ALERT_THRESHOLD:
            self.alert_text = f"ALERT: Yawn ({self.yawn_duration:.1f}s)"
            alert_color = (0, 165, 255)
            current_alert = True

        if current_alert and not self.alert_active:
            self.play_sound_in_thread()
            self.alert_active = True
        elif not current_alert:
            self.alert_active = False

        # --- Display Info ---
        y_offset = 30
        cv2.putText(info_panel, "Drowsiness Monitor", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 40

        if self.alert_text:
            max_len = 35
            if len(self.alert_text) > max_len:
                try:
                    split_point = self.alert_text.rindex(' ', 0, max_len)
                except ValueError:
                    split_point = max_len
                line1 = self.alert_text[:split_point]
                line2 = self.alert_text[split_point:].strip()
                cv2.putText(info_panel, line1, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
                y_offset += 20
                cv2.putText(info_panel, line2, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
            else:
                cv2.putText(info_panel, self.alert_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, alert_color, 1)
            y_offset += 35
        else:
            y_offset += 20

        stats = [
            f"Blinks: {self.blinks}",
            f"Microsleep: {self.microsleep_duration:.2f}s",
            f"Yawns: {self.yawns}",
            f"Yawn Duration: {self.yawn_duration:.2f}s"
        ]

        for stat in stats:
            cv2.putText(info_panel, stat, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30

        try:
            if frame.shape[0] != info_panel.shape[0]:
                info_panel = cv2.resize(info_panel, (info_panel_width, frame.shape[0]))
            return np.hstack((frame, info_panel))
        except Exception as e:
            print(f"Error stacking frame and info panel: {e}")
            return frame

    def capture_frames(self):
        """Continuously captures frames from the webcam."""
        while not self.stop_event.is_set():
            if not self.cap.isOpened():
                print("Capture thread: Webcam closed or not available. Stopping.")
                self.stop_event.set()
                break
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_raw_frame = frame.copy()
            else:
                print("Warning: Failed to grab frame from webcam.")
                time.sleep(0.05)

        print("Capture thread finished.")
        if self.cap.isOpened():
            self.cap.release()

    def process_frames(self):
        """Processes captured frames to detect drowsiness signs."""
        blink_counter_frames = 0

        while not self.stop_event.is_set():
            frame_to_process = None
            with self.frame_lock:
                if self.latest_raw_frame is not None:
                    frame_to_process = self.latest_raw_frame.copy()
                else:
                    time.sleep(0.01)
                    continue

            if frame_to_process is None:
                continue

            # Convert to RGB (MediaPipe expects RGB)
            frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            landmarks = self.process_frame_with_mediapipe(frame_rgb)

            # --- Reset states for this frame ---
            face_detected = False
            left_ear = 1.0
            right_ear = 1.0
            mar = 0.0

            if landmarks is not None and len(landmarks) >= max(
                    max(self.LEFT_EAR_POINTS, self.RIGHT_EAR_POINTS, self.MOUTH_INDICES)):
                face_detected = True

                try:
                    # --- Eye detection (EAR) ---
                    left_eye_coords = landmarks[self.LEFT_EAR_POINTS]
                    right_eye_coords = landmarks[self.RIGHT_EAR_POINTS]
                    left_ear = self.calculate_ear(left_eye_coords)
                    right_ear = self.calculate_ear(right_eye_coords)

                    # --- Yawn detection (MAR) ---
                    mouth_coords = landmarks[self.MOUTH_INDICES]
                    mar = self.calculate_mar(mouth_coords)

                    # Debug prints to verify calculations
                    print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, MAR: {mar:.2f}")

                    # --- Drowsiness Logic ---
                    current_left_eye_closed = left_ear < self.EAR_THRESHOLD
                    current_right_eye_closed = right_ear < self.EAR_THRESHOLD

                    # -- Blink/Microsleep Logic --
                    if current_left_eye_closed and current_right_eye_closed:
                        if not (self.left_eye_still_closed and self.right_eye_still_closed):
                            # Start of closure
                            self.left_eye_still_closed = True
                            self.right_eye_still_closed = True
                            self.microsleep_start_time = time.time()
                            blink_counter_frames = 1
                        else:
                            # Closure continues
                            blink_counter_frames += 1
                            if self.microsleep_start_time:
                                self.microsleep_duration = time.time() - self.microsleep_start_time
                            else:
                                self.microsleep_start_time = time.time()
                                self.microsleep_duration = 0.0
                    else:
                        # At least one eye is open
                        if self.left_eye_still_closed or self.right_eye_still_closed:
                            # Eyes just opened
                            if blink_counter_frames > 0:
                                self.blinks += 1
                                print(f"Blink detected! Total blinks: {self.blinks}")

                        # Reset closed state and microsleep timer/duration
                        self.left_eye_still_closed = False
                        self.right_eye_still_closed = False
                        self.microsleep_start_time = None
                        self.microsleep_duration = 0.0
                        blink_counter_frames = 0

                    # -- Yawn Logic --
                    if mar > self.MAR_THRESHOLD:
                        self.yawn_frames += 1
                        if self.yawn_frames >= self.MIN_YAWN_FRAMES and not self.yawn_in_progress:
                            # Yawn Start
                            self.yawns += 1
                            self.yawn_in_progress = True
                            self.yawn_start_time = time.time()
                            self.yawn_duration = 0.0
                            print(f"Yawn detected! MAR: {mar:.2f}")

                        if self.yawn_in_progress and self.yawn_start_time is not None:
                            self.yawn_duration = time.time() - self.yawn_start_time
                    else:
                        if self.yawn_in_progress:
                            final_duration = self.yawn_duration
                            print(f"Yawn ended. Duration: {self.yawn_duration:.2f}s")

                        # Reset yawn state
                        self.yawn_frames = 0
                        self.yawn_in_progress = False
                        self.yawn_start_time = None
                        self.yawn_duration = 0.0

                except IndexError as e:
                    print(f"Landmark index error: {e}")
                    face_detected = False

            # --- Handle No Face Detected ---
            if not face_detected:
                # Reset all states if no face detected
                self.left_eye_still_closed = False
                self.right_eye_still_closed = False
                self.microsleep_start_time = None
                self.microsleep_duration = 0.0
                self.yawn_in_progress = False
                self.yawn_start_time = None
                self.yawn_duration = 0.0
                self.yawn_frames = 0
                blink_counter_frames = 0

            # --- Update Processed Frame ---
            processed_frame_with_info = self.update_info(frame_to_process)
            with self.frame_lock:
                self.latest_processed_frame = processed_frame_with_info

        print("Process thread finished.")
        self.landmarker.close()

    def display_frames(self):
        """Displays the processed frames with drowsiness information."""
        while not self.stop_event.is_set():
            display_frame = None
            with self.frame_lock:
                if self.latest_processed_frame is not None:
                    display_frame = self.latest_processed_frame.copy()
                elif self.latest_raw_frame is not None:
                    display_frame = self.update_info(self.latest_raw_frame.copy())
                else:
                    time.sleep(0.01)
                    continue

            if display_frame is not None:
                cv2.imshow('Drowsiness Detection', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("Quit key pressed.")
                self.stop_event.set()
                break

        print("Display thread finished. Cleaning up windows...")
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)

    def play_alert_sound(self):
        """Plays a system beep sound (Windows only by default)."""
        if SOUND_AVAILABLE:
            try:
                winsound.Beep(1000, 500)
            except Exception as e:
                print(f"Error playing sound: {e}")

    def play_sound_in_thread(self):
        """Plays the alert sound in a separate thread to avoid blocking."""
        if SOUND_AVAILABLE:
            sound_thread = threading.Thread(target=self.play_alert_sound)
            sound_thread.daemon = True
            sound_thread.start()

    def run(self):
        """Keeps the main thread alive and handles cleanup."""
        try:
            while not self.stop_event.is_set():
                if not self.capture_thread.is_alive() or \
                        not self.process_thread.is_alive() or \
                        not self.display_thread.is_alive():
                    print("Error: An essential thread has stopped unexpectedly. Stopping application.")
                    self.stop_event.set()
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Stopping threads...")
            self.stop_event.set()
        finally:
            print("Waiting for threads to join...")
            if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
            if hasattr(self, 'process_thread') and self.process_thread.is_alive():
                self.process_thread.join(timeout=2.0)
            if hasattr(self, 'display_thread') and self.display_thread.is_alive():
                self.display_thread.join(timeout=2.0)

            print("All threads should have finished.")
            if self.cap.isOpened():
                self.cap.release()
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)


if __name__ == "__main__":
    print("Starting Drowsiness Detector with MediaPipe...")
    detector = DrowsinessDetector()
    if not detector.stop_event.is_set():
        detector.run()
    else:
        print("Drowsiness detector failed to initialize properly.")
    print("Application Finished.")