import cv2
import mediapipe as mp
import time
import numpy as np
import datetime

class PoseTracker:
    def __init__(self, camera_index=1):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Model Configuration
        self.model_complexity = 2
        self.min_detection_confidence = 0.7
        self.min_tracking_confidence = 0.7
        
        self.pose = self.mp_pose.Pose(
            model_complexity=self.model_complexity, 
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)
        
        # Resolution and Performance Settings
        self.width = 1280
        self.height = 720
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.prev_time = time.time()
        self.last_results = None
        self.show_debug = False
        self.latency = 0
        
        # Recording Settings
        self.is_recording = False
        self.video_writer = None
        self.recording_filename = ""

    def toggle_recording(self):
        """
        Starts or stops the video recording.
        """
        if not self.is_recording:
            # Start Recording
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.recording_filename = f"pose_record_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.recording_filename, fourcc, 30.0, (self.width, self.height))
            self.is_recording = True
            print(f"Started recording: {self.recording_filename}")
        else:
            # Stop Recording
            self.is_recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            print(f"Stopped recording. Saved to: {self.recording_filename}")

    def draw_ui(self, frame, fps):
        """
        UI with FPS, Recording Status, and optional Debug Screen.
        """
        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_DUPLEX
        
        # Top Bar for Status
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Left: FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), font, 1.0, (0, 255, 0), 2)
        
        # Center: Recording Indicator
        if self.is_recording:
            # Pulsing red dot effect
            if int(time.time() * 2) % 2 == 0:
                cv2.circle(frame, (w // 2 - 100, 35), 10, (0, 0, 255), -1)
            cv2.putText(frame, "RECORDING", (w // 2 - 80, 45), font, 1.0, (0, 0, 255), 2)
        
        # Right: Instructions
        cv2.putText(frame, "R: Record | D: Debug | ESC: Exit", (w - 480, 40), font, 0.7, (200, 200, 200), 1)

        if self.show_debug:
            self.draw_debug_screen(frame)

    def draw_debug_screen(self, frame):
        """
        Draws a detailed debug overlay on the right side of the screen.
        """
        h, w, _ = frame.shape
        debug_w = 400
        debug_h = 350
        
        # Debug Panel Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - debug_w - 10, 70), (w - 10, 70 + debug_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        x_offset = w - debug_w
        y_start = 110
        line_height = 35
        
        debug_info = [
            ("DEBUG MODE", (0, 255, 255), 0.8, 2),
            ("-" * 20, (150, 150, 150), 0.5, 1),
            (f"Model Complexity: {self.model_complexity}", (255, 255, 255), 0.6, 1),
            (f"Min Detect Conf: {self.min_detection_confidence}", (255, 255, 255), 0.6, 1),
            (f"Latency: {self.latency:.2f} ms", (0, 255, 0) if self.latency < 50 else (0, 165, 255), 0.6, 1),
            (f"Resolution: {self.width}x{self.height}", (255, 255, 255), 0.6, 1),
            (f"Landmarks: {'Detected' if self.last_results and self.last_results.pose_landmarks else 'None'}", (255, 255, 255), 0.6, 1),
            (f"Recording: {'ON' if self.is_recording else 'OFF'}", (0, 0, 255) if self.is_recording else (255, 255, 255), 0.6, 1),
            (f"Camera Index: {self.camera_index}", (255, 255, 255), 0.6, 1),
        ]
        
        for i, (text, color, scale, thick) in enumerate(debug_info):
            cv2.putText(frame, text, (x_offset, y_start + i * line_height), font, scale, color, thick)

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break

            # Performance tracking
            curr_time = time.time()
            dt = curr_time - self.prev_time
            self.prev_time = curr_time
            fps = 1/dt if dt > 0 else 0
            
            # Pre-process frame
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Model Processing with Latency Measurement
            start_proc = time.time()
            self.last_results = self.pose.process(img_rgb)
            self.latency = (time.time() - start_proc) * 1000 # Convert to ms
            
            if self.last_results and self.last_results.pose_landmarks:
                # Draw skeleton
                self.mp_drawing.draw_landmarks(
                    frame, self.last_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )

            # Draw UI and Debug Screen
            self.draw_ui(frame, fps)

            # Write frame to video if recording
            if self.is_recording and self.video_writer:
                self.video_writer.write(frame)

            cv2.imshow('Pose Tracker Debug & Record', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                break
            elif key == ord('d') or key == ord('D'):
                self.show_debug = not self.show_debug
            elif key == ord('r') or key == ord('R'):
                self.toggle_recording()

        # Cleanup
        if self.video_writer:
            self.video_writer.release()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use camera index 0 or 1 depending on your setup
    app = PoseTracker(camera_index=1)
    app.run()
