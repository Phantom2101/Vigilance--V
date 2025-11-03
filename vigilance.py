# vigilance.py
import sys
import os
import cv2
import time
import csv
from datetime import datetime
from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path
from face_manager import (get_face_detector, save_face_image,
                          train_lbph_model, load_model, list_registered_people, DATASET_DIR)

# Ensure logs folder
Path("logs").mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("logs/dwell_log.csv")

class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(object)

    def __init__(self, src=0):
        super().__init__()
        self._run_flag = False
        self.src = src
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.src)
        # small warm-up
        time.sleep(0.5)
        self._run_flag = True
        while self._run_flag:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.change_pixmap_signal.emit(frame)
            # limit frame rate a little
            self.msleep(20)
        if self.cap:
            self.cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vigilance - Smart Surveillance")
        self.setGeometry(100, 100, 900, 600)
        self.detector = get_face_detector()
        self.recognizer, self.label_map = load_model()
        self.dwell_tracker = {}  # label -> start_time (datetime)
        self.currently_seen = set()  # labels currently present this frame
        self.recognition_threshold = 60.0  # lower is stricter
        self.capturing_registration = False
        self.registration_name = None
        self.registration_count = 0
        self.registration_target = 30  # images per person

        self._build_ui()

        # thread
        self.thread = VideoThread(src=0)
        self.thread.change_pixmap_signal.connect(self.process_frame)

    def _build_ui(self):
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)
        layout = QtWidgets.QHBoxLayout()
        widget.setLayout(layout)

        # Left: video display
        left = QtWidgets.QVBoxLayout()
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(640, 480)
        self.video_label.setStyleSheet("background-color: #222;")
        left.addWidget(self.video_label)

        # Info labels
        self.info_label = QtWidgets.QLabel("Status: Idle")
        left.addWidget(self.info_label)

        layout.addLayout(left)

        # Right: controls
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

        self.register_btn = QtWidgets.QPushButton("Start Registration (capture 30 images)")
        self.register_btn.clicked.connect(self.start_registration)
        reg_layout.addWidget(self.register_btn)

        self.capture_progress = QtWidgets.QLabel("Progress: 0/30")
        reg_layout.addWidget(self.capture_progress)

        right.addWidget(reg_box)

        # Training
        self.train_btn = QtWidgets.QPushButton("Train Model")
        self.train_btn.clicked.connect(self.train_model)
        right.addWidget(self.train_btn)

        # Registered people list
        self.people_list = QtWidgets.QListWidget()
        self.refresh_people_list()
        right.addWidget(QtWidgets.QLabel("Registered People:"))
        right.addWidget(self.people_list)

        # Logs
        self.view_logs_btn = QtWidgets.QPushButton("View Logs (CSV)")
        self.view_logs_btn.clicked.connect(self.open_logs)
        right.addWidget(self.view_logs_btn)

        # Spacer and exit
        right.addStretch()
        self.exit_btn = QtWidgets.QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)
        right.addWidget(self.exit_btn)

        layout.addLayout(right)

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
        # ensure thread stops
        if self.thread.isRunning():
            self.thread.stop()
        event.accept()

    def start_registration(self):
        name = self.name_input.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Name required", "Please enter a name for registration.")
            return
        # sanitize: no spaces
        name = name.replace(" ", "_")
        self.registration_name = name
        self.capturing_registration = True
        self.registration_count = 0
        self.capture_progress.setText(f"Progress: {self.registration_count}/{self.registration_target}")
        self.info_label.setText(f"Status: Registering {name} - look at the camera")
        # create folder if not exists
        (DATASET_DIR / name).mkdir(parents=True, exist_ok=True)

    def train_model(self):
        try:
            train_lbph_model()
            self.recognizer, self.label_map = load_model()
            QtWidgets.QMessageBox.information(self, "Training complete", "LBPH model trained and saved.")
            self.refresh_people_list()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Training error", str(e))

    def refresh_people_list(self):
        self.people_list.clear()
        for p in list_registered_people():
            self.people_list.addItem(p)

    def open_logs(self):
        # open CSV using default application or show popup path
        if not LOG_FILE.exists():
            QtWidgets.QMessageBox.information(self, "Logs", "No logs yet.")
            return
        # try to open with OS default
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(LOG_FILE))
            elif sys.platform == "darwin":
                os.system(f"open {LOG_FILE}")
            else:
                os.system(f"xdg-open {LOG_FILE}")
        except Exception:
            QtWidgets.QMessageBox.information(self, "Logs", f"Log file at: {LOG_FILE}")

    def process_frame(self, frame):
        # frame: BGR numpy array
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        new_seen = set()
        display_frame = frame.copy()

        # registration capture if active
        if self.capturing_registration and len(faces) > 0:
            # find largest face and capture
            largest = max(faces, key=lambda r: r[2]*r[3])
            (x, y, w, h) = largest
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (200, 200))
            save_face_image(self.registration_name, face_resized, self.registration_count)
            self.registration_count += 1
            self.capture_progress.setText(f"Progress: {self.registration_count}/{self.registration_target}")
            # draw rectangle
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,255), 2)
            cv2.putText(display_frame, f"Capturing {self.registration_name}: {self.registration_count}/{self.registration_target}",
                        (10, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            if self.registration_count >= self.registration_target:
                self.capturing_registration = False
                self.registration_name = None
                self.info_label.setText("Status: Registration completed. Please Train Model.")
                QtWidgets.QMessageBox.information(self, "Registration", "Image capture complete. Click Train Model.")
                self.refresh_people_list()

        # recognition logic (only if recognizer loaded)
        if self.recognizer is not None:
            for (x, y, w, h) in faces:
                face_img = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_img, (200, 200))
                label, confidence = self.recognizer.predict(face_resized)
                name = self.label_map.get(label, "Unknown")
                # confidence: lower = better match for LBPH
                if confidence < self.recognition_threshold:
                    # recognized
                    new_seen.add(label)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255,0,0), 2)
                    cv2.putText(display_frame, f"{name} ({int(confidence)})", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
                else:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    cv2.putText(display_frame, "Unknown", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            # no model: only draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # dwell time tracking:
        # Start timers for newly seen labels
        for lbl in new_seen:
            if lbl not in self.dwell_tracker:
                self.dwell_tracker[lbl] = datetime.now()
        # Check labels that disappeared
        previous_seen = set(self.dwell_tracker.keys()) & (self.currently_seen if self.currently_seen else set())
        # Actually better logic: compare last frame seen set to new_seen
        lost = (self.currently_seen - new_seen) if self.currently_seen else set()
        for lbl in lost:
            start = self.dwell_tracker.pop(lbl, None)
            if start:
                duration = (datetime.now() - start).total_seconds()
                name = self.label_map.get(lbl, f"label_{lbl}")
                self.log_dwell(name, duration)
                # small visual indicator on frame
                cv2.putText(display_frame, f"{name} left: {int(duration)}s", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # update current seen
        self.currently_seen = new_seen

        # Show live dwell times on top-left
        y0 = 30
        for i, (lbl, start) in enumerate(self.dwell_tracker.items()):
            name = self.label_map.get(lbl, f"label_{lbl}")
            live_seconds = int((datetime.now() - start).total_seconds())
            cv2.putText(display_frame, f"{name}: {live_seconds}s", (10, y0 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # convert to Qt format and show
        rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt_image).scaled(self.video_label.width(), self.video_label.height(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)

    def log_dwell(self, name, duration_seconds):
        # append to CSV with timestamp
        header = ["timestamp", "person", "duration_seconds"]
        exists = LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(header)
            writer.writerow([datetime.now().isoformat(), name, int(duration_seconds)])
        print(f"Logged: {name}, {duration_seconds}s")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
