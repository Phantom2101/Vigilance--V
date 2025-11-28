# vigilance.py
import sys
import os
import cv2
import time
import csv
import tempfile
import numpy as np
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path
from deepface import DeepFace

from face_manager import (
    save_face_image,
    train_deepface_model,
    list_registered_people,
    DATASET_DIR
)

# Ensure logs folder
Path("logs").mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("logs/dwell_log.csv")


# =======================
# Video Thread
# =======================
class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(object)

    def __init__(self, src=0):
        super().__init__()
        self._run_flag = False
        self.src = src
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.src)
        time.sleep(0.5)
        self._run_flag = True
        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.change_pixmap_signal.emit(frame)
            self.msleep(20)
        if self.cap:
            self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()


# =======================
# Main Window Class
# =======================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vigilance - Smart Surveillance (DeepFace)")
        self.setGeometry(100, 100, 900, 600)

        self.dwell_tracker = {}  # name -> start_time
        self.currently_seen = set()
        self.capturing_registration = False
        self.registration_name = None
        self.registration_count = 0
        self.registration_target = 30
        self.recognition_threshold = 0.5  # Facenet distance threshold

        self._build_ui()

        # Start camera thread
        self.thread = VideoThread(src=0)
        self.thread.change_pixmap_signal.connect(self.process_frame)

    # -----------------------
    # UI Setup
    # -----------------------
    def _build_ui(self):
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QHBoxLayout()
        widget.setLayout(layout)

        # Left: Video Display
        left = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #222;")
        left.addWidget(self.video_label)
        self.info_label = QtWidgets.QLabel("Status: Idle")
        left.addWidget(self.info_label)
        layout.addLayout(left)

        # Right: Controls
        right = QtWidgets.QVBoxLayout()

        self.start_btn = QtWidgets.QPushButton("Start Camera")
        self.start_btn.clicked.connect(self.start_camera)
        right.addWidget(self.start_btn)

        self.stop_btn = QtWidgets.QPushButton("Stop Camera")
        self.stop_btn.clicked.connect(self.stop_camera)
        self.stop_btn.setEnabled(False)
        right.addWidget(self.stop_btn)

        # Registration
        reg_box = QtWidgets.QGroupBox("Face Registration")
        reg_layout = QtWidgets.QVBoxLayout()
        reg_box.setLayout(reg_layout)

        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("Enter person name (no spaces)")
        reg_layout.addWidget(self.name_input)

        self.register_btn = QtWidgets.QPushButton("Start Registration (30 images)")
        self.register_btn.clicked.connect(self.start_registration)
        reg_layout.addWidget(self.register_btn)

        self.capture_progress = QtWidgets.QLabel("Progress: 0/30")
        reg_layout.addWidget(self.capture_progress)
        right.addWidget(reg_box)

        # Train button
        self.train_btn = QtWidgets.QPushButton("Build Face Database (DeepFace)")
        self.train_btn.clicked.connect(self.train_model)
        right.addWidget(self.train_btn)

        # Registered list
        self.people_list = QtWidgets.QListWidget()
        self.refresh_people_list()
        right.addWidget(QtWidgets.QLabel("Registered People:"))
        right.addWidget(self.people_list)

        # Logs and Exit
        self.view_logs_btn = QtWidgets.QPushButton("View Logs (CSV)")
        self.view_logs_btn.clicked.connect(self.open_logs)
        right.addWidget(self.view_logs_btn)

        right.addStretch()
        self.exit_btn = QtWidgets.QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        right.addWidget(self.exit_btn)

        layout.addLayout(right)

    # -----------------------
    # Camera Control
    # -----------------------
    def start_camera(self):
        if not self.thread.isRunning():
            self.thread.start()
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.info_label.setText("Status: Camera running")
        else:
            self.info_label.setText("Status: Already running")

    def stop_camera(self):
        if self.thread.isRunning():
            self.thread.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.info_label.setText("Status: Camera stopped")

    def closeEvent(self, event):
        if self.thread.isRunning():
            self.thread.stop()
        event.accept()

    # -----------------------
    # Registration & Training
    # -----------------------
    def start_registration(self):
        name = self.name_input.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Name required", "Enter a name to register.")
            return
        name = name.replace(" ", "_")
        self.registration_name = name
        self.capturing_registration = True
        self.registration_count = 0
        self.capture_progress.setText(f"Progress: 0/{self.registration_target}")
        self.info_label.setText(f"Status: Registering {name}. Move head slowly...")
        (DATASET_DIR / name).mkdir(parents=True, exist_ok=True)

    def train_model(self):
        try:
            train_deepface_model()
            QtWidgets.QMessageBox.information(
                self, "Database Ready", "Face embeddings built successfully."
            )
            self.refresh_people_list()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", str(e))

    def refresh_people_list(self):
        self.people_list.clear()
        for p in list_registered_people():
            self.people_list.addItem(p)

    # -----------------------
    # Logs
    # -----------------------
    def open_logs(self):
        if not LOG_FILE.exists():
            QtWidgets.QMessageBox.information(self, "Logs", "No logs yet.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(LOG_FILE))
            elif sys.platform == "darwin":
                os.system(f"open {LOG_FILE}")
            else:
                os.system(f"xdg-open {LOG_FILE}")
        except Exception:
            QtWidgets.QMessageBox.information(self, "Logs", f"Log file: {LOG_FILE}")

    # -----------------------
    # Frame Processing (DeepFace Recognition)
    # -----------------------
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        display_frame = frame.copy()

        # Face detection using OpenCV (faster than DeepFace for detection only)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(rgb_frame, 1.2, 5, minSize=(60, 60))

        new_seen = set()

        # Registration Mode
        if self.capturing_registration and len(faces) > 0:
            x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
            face_img = frame[y:y+h, x:x+w]
            save_face_image(self.registration_name, face_img, self.registration_count)
            self.registration_count += 1
            self.capture_progress.setText(f"Progress: {self.registration_count}/{self.registration_target}")
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,255), 2)
            if self.registration_count >= self.registration_target:
                self.capturing_registration = False
                self.info_label.setText("Status: Registration complete. Build database next.")
                QtWidgets.QMessageBox.information(self, "Done", "Registration complete.")
                self.refresh_people_list()

        # Recognition Mode
        elif len(faces) > 0:
            for (x, y, w, h) in faces:
                face_crop = frame[y:y+h, x:x+w]
                temp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(temp_img.name, face_crop)
                try:
                    result = DeepFace.find(
                        img_path=temp_img.name,
                        db_path=str(DATASET_DIR),
                        model_name="Facenet",
                        enforce_detection=False,
                        silent=True
                    )
                    if not result.empty:
                        best_match = result.iloc[0]
                        name = Path(best_match['identity']).parent.name
                        distance = best_match['distance']
                        if distance < self.recognition_threshold:
                            new_seen.add(name)
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255,0,0), 2)
                            cv2.putText(display_frame, f"{name} ({distance:.2f})",
                                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
                        else:
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)
                            cv2.putText(display_frame, "Unknown", (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                except Exception as e:
                    print("Recognition error:", e)

        # Dwell Tracking
        for name in new_seen:
            if name not in self.dwell_tracker:
                self.dwell_tracker[name] = datetime.now()
        lost = self.currently_seen - new_seen if self.currently_seen else set()
        for name in lost:
            start = self.dwell_tracker.pop(name, None)
            if start:
                duration = (datetime.now() - start).total_seconds()
                self.log_dwell(name, duration)
                cv2.putText(display_frame, f"{name} left ({int(duration)}s)",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        self.currently_seen = new_seen

        # Show live dwell timers
        y0 = 30
        for i, (name, start) in enumerate(self.dwell_tracker.items()):
            live_seconds = int((datetime.now() - start).total_seconds())
            cv2.putText(display_frame, f"{name}: {live_seconds}s",
                        (10, y0 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # Display frame
        rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_display.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_display.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt_image).scaled(self.video_label.width(),
                                                       self.video_label.height(),
                                                       QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    # -----------------------
    # Log Dwell Times
    # -----------------------
    def log_dwell(self, name, duration_seconds):
        header = ["timestamp", "person", "duration_seconds"]
        exists = LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(header)
            writer.writerow([datetime.now().isoformat(), name, int(duration_seconds)])
        print(f"[LOG] {name} stayed {int(duration_seconds)}s")


# =======================
# Entry Point
# =======================
def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
