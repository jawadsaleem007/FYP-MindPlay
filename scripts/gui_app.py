"""Desktop GUI for EEG FBCSP+LDA training and real-time classification.

Features:
- Main menu with:
  - Start Training
  - Real Time Classification
  - Exit
- Training pipeline:
  1) Record trials from LSL
  2) Train FBCSP+LDA model
  3) Evaluate model and optionally retry if CV accuracy < 60%
- Real-time classification view with live console output and Stop control.

Run:
  python .\scripts\gui_app.py
"""
from __future__ import annotations

import re
import sys
import subprocess
import queue
from pathlib import Path
from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QTextEdit,
    QMessageBox,
    QStackedWidget,
    QFrame,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA_DIR = ROOT / "data"


class TrainingWorker(QThread):
    log_line = pyqtSignal(str)
    status = pyqtSignal(str)
    error = pyqtSignal(str, str)
    low_accuracy = pyqtSignal(float)
    finished_ok = pyqtSignal(str)

    def __init__(self, subject: str) -> None:
        super().__init__()
        self.subject = subject
        self._decision_queue: queue.Queue[bool] = queue.Queue(maxsize=1)

    def set_retry_decision(self, retry: bool) -> None:
        try:
            self._decision_queue.put_nowait(retry)
        except queue.Full:
            pass

    @staticmethod
    def _run_cmd_stream(cmd: list[str], cwd: Path, emit_fn) -> Tuple[int, str]:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines = []
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\r\n")
            lines.append(line)
            emit_fn(line)
        proc.wait()
        return int(proc.returncode), "\n".join(lines)

    @staticmethod
    def _extract_saved_paths(log_text: str) -> Tuple[Optional[Path], Optional[Path]]:
        ep = None
        lb = None
        m1 = re.findall(r"([\\/\w\-\.]+_epochs_\d{8}_\d{6}\.npy)", log_text)
        m2 = re.findall(r"([\\/\w\-\.]+_labels_\d{8}_\d{6}\.npy)", log_text)
        if m1:
            ep = Path(m1[-1])
        if m2:
            lb = Path(m2[-1])
        return ep, lb

    @staticmethod
    def _latest_subject_files(subject: str) -> Tuple[Optional[Path], Optional[Path]]:
        ep_candidates = sorted(DATA_DIR.glob(f"{subject}_epochs_*.npy"))
        lb_candidates = sorted(DATA_DIR.glob(f"{subject}_labels_*.npy"))
        ep = ep_candidates[-1] if ep_candidates else None
        lb = lb_candidates[-1] if lb_candidates else None
        return ep, lb

    @staticmethod
    def _parse_cv_accuracy_percent(eval_text: str) -> Optional[float]:
        m = re.search(r"CV\s+\d+\-fold\s+accuracy:\s*([0-9]*\.?[0-9]+)", eval_text, flags=re.IGNORECASE)
        if not m:
            return None
        val = float(m.group(1))
        return val * 100.0 if val <= 1.0 else val

    def run(self) -> None:
        try:
            while True:
                self.status.emit("Step 1/3: Recording trials from LSL...")
                record_cmd = [
                    sys.executable,
                    str(SCRIPTS / "record_trials_lsl.py"),
                    "--subject",
                    self.subject,
                    "--picks",
                    "C3,Cz,C4",
                    "--trial-len",
                    "4.0",
                    "--trials-per-class",
                    "25",
                    "--prep-len",
                    "2.0",
                    "--inter-trial",
                    "2.0",
                    "--randomize",
                    "--scale-to-uv",
                ]
                rc, rec_log = self._run_cmd_stream(record_cmd, ROOT, self.log_line.emit)
                if rc != 0:
                    self.error.emit("Recording Failed", "Step 1 failed. Check LSL stream and try again.")
                    return

                epochs_path, labels_path = self._extract_saved_paths(rec_log)
                if epochs_path is None or labels_path is None:
                    epochs_path, labels_path = self._latest_subject_files(self.subject)
                if epochs_path is None or labels_path is None:
                    self.error.emit("Files Not Found", "Could not locate saved epochs/labels in data folder.")
                    return

                if not epochs_path.is_absolute():
                    epochs_path = (ROOT / epochs_path).resolve()
                if not labels_path.is_absolute():
                    labels_path = (ROOT / labels_path).resolve()

                model_path = (ROOT / f"fbcsp_lda_{self.subject}.joblib").resolve()

                self.status.emit("Step 2/3: Its Training Wait Please")
                train_cmd = [
                    sys.executable,
                    str(SCRIPTS / "train_fbcsp_lda.py"),
                    "--epochs",
                    str(epochs_path),
                    "--labels",
                    str(labels_path),
                    "--sfreq",
                    "500.0",
                    "--out",
                    str(model_path),
                ]
                rc, _ = self._run_cmd_stream(train_cmd, ROOT, self.log_line.emit)
                if rc != 0:
                    self.error.emit("Training Failed", "Step 2 failed. Check logs and retry.")
                    return

                self.status.emit("Step 3/3: Evaluating trained model...")
                eval_cmd = [
                    sys.executable,
                    str(SCRIPTS / "evaluate_trained_model.py"),
                    "--model",
                    str(model_path),
                    "--epochs",
                    str(epochs_path),
                    "--labels",
                    str(labels_path),
                    "--sfreq",
                    "500",
                    "--folds",
                    "5",
                    "--picks",
                    "0,1,2",
                ]
                rc, eval_log = self._run_cmd_stream(eval_cmd, ROOT, self.log_line.emit)
                if rc != 0:
                    self.error.emit("Evaluation Failed", "Step 3 failed. Check logs and retry.")
                    return

                acc = self._parse_cv_accuracy_percent(eval_log)
                if acc is not None:
                    self.status.emit(f"Evaluation complete: CV accuracy = {acc:.2f}%")
                else:
                    self.status.emit("Evaluation complete.")

                if acc is not None and acc < 60.0:
                    self.low_accuracy.emit(acc)
                    retry = self._decision_queue.get()
                    if retry:
                        self.log_line.emit("User selected RETRY. Restarting from Step 1...")
                        continue

                self.finished_ok.emit(self.subject)
                return

        except Exception as e:
            self.error.emit("Unexpected Error", str(e))


class EEGApp(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EEG MI Pipeline - Modern DS4")
        self.resize(1180, 790)
        self.setMinimumSize(980, 680)

        self.training_worker: Optional[TrainingWorker] = None
        self.rt_process: Optional[QProcess] = None

        self._build_ui()
        self._apply_theme()

    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)
        root = QVBoxLayout(container)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        self.stack = QStackedWidget()
        root.addWidget(self.stack)

        self.menu_page = self._create_menu_page()
        self.train_page = self._create_training_page()
        self.rt_page = self._create_rt_page()

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.rt_page)
        self.show_menu()

    def _title_label(self, text: str) -> QLabel:
        lb = QLabel(text)
        lb.setObjectName("title")
        lb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lb

    def _card(self) -> QFrame:
        fr = QFrame()
        fr.setObjectName("card")
        return fr

    def _create_menu_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(16)

        lay.addWidget(self._title_label("EEG MI Control Panel"))


        card = self._card()
        card_lay = QVBoxLayout(card)
        card_lay.setContentsMargins(26, 24, 26, 24)
        card_lay.setSpacing(14)

        h = QLabel("Main Menu")
        h.setObjectName("cardTitle")
        card_lay.addWidget(h)

        row = QHBoxLayout()
        row.setSpacing(10)
        btn_train = QPushButton("Start Training")
        btn_train.setObjectName("primary")
        btn_train.clicked.connect(self.show_training)
        btn_rt = QPushButton("Real Time Classification")
        btn_rt.setObjectName("primary")
        btn_rt.clicked.connect(self.show_realtime)
        btn_exit = QPushButton("Exit")
        btn_exit.setObjectName("ghost")
        btn_exit.clicked.connect(self.close)
        row.addWidget(btn_train)
        row.addWidget(btn_rt)
        row.addWidget(btn_exit)
        card_lay.addLayout(row)

        lay.addWidget(card)
        lay.addStretch(1)
        return page

    def _create_training_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(10)
        lay.addWidget(self._title_label("Start Training"))

        top = self._card()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(16, 14, 16, 14)
        top_lay.setSpacing(10)
        top_lay.addWidget(QLabel("Subject Name:"))
        self.subject_input = QLineEdit("S01")
        self.subject_input.setFixedWidth(220)
        top_lay.addWidget(self.subject_input)
        self.btn_train_start = QPushButton("Run Full Training")
        self.btn_train_start.setObjectName("primary")
        self.btn_train_start.clicked.connect(self.start_training_pipeline)
        top_lay.addWidget(self.btn_train_start)
        btn_back = QPushButton("Back to Menu")
        btn_back.setObjectName("ghost")
        btn_back.clicked.connect(self.show_menu)
        top_lay.addWidget(btn_back)
        top_lay.addStretch(1)
        lay.addWidget(top)

        self.train_status = QLabel("Ready")
        self.train_status.setObjectName("status")
        lay.addWidget(self.train_status)

        self.big_prompt = QLabel("")
        self.big_prompt.setObjectName("bigPrompt")
        self.big_prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.big_prompt)

        log_card = self._card()
        log_lay = QVBoxLayout(log_card)
        log_lay.setContentsMargins(12, 12, 12, 12)
        cap = QLabel("Pipeline Output")
        cap.setObjectName("cardTitle")
        log_lay.addWidget(cap)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setObjectName("console")
        log_lay.addWidget(self.train_log)
        lay.addWidget(log_card, 1)
        return page

    def _create_rt_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setSpacing(10)
        lay.addWidget(self._title_label("Real Time Classification"))

        top = self._card()
        top_lay = QHBoxLayout(top)
        top_lay.setContentsMargins(16, 14, 16, 14)
        top_lay.setSpacing(10)
        top_lay.addWidget(QLabel("Subject Name:"))
        self.rt_subject_input = QLineEdit("S01")
        self.rt_subject_input.setFixedWidth(220)
        top_lay.addWidget(self.rt_subject_input)
        self.btn_rt_start = QPushButton("Start Real Time")
        self.btn_rt_start.setObjectName("primary")
        self.btn_rt_start.clicked.connect(self.start_realtime)
        top_lay.addWidget(self.btn_rt_start)
        self.btn_rt_stop = QPushButton("Stop")
        self.btn_rt_stop.setObjectName("ghost")
        self.btn_rt_stop.clicked.connect(self.stop_realtime)
        top_lay.addWidget(self.btn_rt_stop)
        btn_back = QPushButton("Back to Menu")
        btn_back.setObjectName("ghost")
        btn_back.clicked.connect(self.back_from_rt)
        top_lay.addWidget(btn_back)
        top_lay.addStretch(1)
        lay.addWidget(top)

        self.rt_status = QLabel("Ready")
        self.rt_status.setObjectName("status")
        lay.addWidget(self.rt_status)

        log_card = self._card()
        log_lay = QVBoxLayout(log_card)
        log_lay.setContentsMargins(12, 12, 12, 12)
        cap = QLabel("Classifier Output")
        cap.setObjectName("cardTitle")
        log_lay.addWidget(cap)
        self.rt_log = QTextEdit()
        self.rt_log.setReadOnly(True)
        self.rt_log.setObjectName("console")
        log_lay.addWidget(self.rt_log)
        lay.addWidget(log_card, 1)
        return page

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #dff3ff; color: #0f2b3d; font-family: 'Segoe UI'; font-size: 14px; }
            QLabel#title { font-size: 34px; font-weight: 800; color: #0e2a3a; padding: 4px; }
            QLabel#subtitle { font-size: 15px; color: #3d6882; margin-bottom: 8px; }
            QFrame#card { background-color: #f8fcff; border: 1px solid #cce8f9; border-radius: 16px; }
            QLabel#cardTitle { font-size: 18px; font-weight: 700; color: #123a52; }
            QPushButton { border-radius: 12px; padding: 10px 14px; font-size: 14px; font-weight: 700; min-height: 22px; }
            QPushButton#primary { background: #1f94df; color: white; border: none; }
            QPushButton#primary:hover { background: #1982c5; }
            QPushButton#primary:pressed { background: #136c9f; }
            QPushButton#ghost { background: white; color: #0f6ca6; border: 1px solid #8cc7eb; }
            QPushButton#ghost:hover { background: #edf7ff; }
            QLineEdit { background: white; border: 1px solid #9fd2ef; border-radius: 10px; padding: 8px 10px; color: #15394f; }
            QLabel#status { background-color: #ecf7ff; border: 1px solid #cce7f7; border-radius: 10px; padding: 10px 12px; font-weight: 700; color: #0f6ca6; }
            QLabel#bigPrompt { color: #0d7a34; font-size: 40px; font-weight: 900; padding: 6px; }
            QTextEdit#console { background: white; border: 1px solid #c6e6fa; border-radius: 10px; color: #123448; font-family: Consolas; font-size: 12px; }
            """
        )

    def show_menu(self) -> None:
        self.stack.setCurrentWidget(self.menu_page)

    def show_training(self) -> None:
        self.stack.setCurrentWidget(self.train_page)

    def show_realtime(self) -> None:
        self.stack.setCurrentWidget(self.rt_page)

    def append_train_log(self, text: str) -> None:
        self.train_log.append(text)

    def append_rt_log(self, text: str) -> None:
        self.rt_log.append(text)

    def start_training_pipeline(self) -> None:
        subject = self.subject_input.text().strip()
        if not subject:
            QMessageBox.warning(self, "Missing Subject", "Please enter subject name.")
            return
        if self.training_worker and self.training_worker.isRunning():
            QMessageBox.information(self, "Training Running", "A training session is already running.")
            return

        self.btn_train_start.setEnabled(False)
        self.big_prompt.setText("")
        self.append_train_log(f"\n=== New training session for {subject} ===")

        self.training_worker = TrainingWorker(subject)
        self.training_worker.log_line.connect(self.append_train_log)
        self.training_worker.status.connect(self.train_status.setText)
        self.training_worker.error.connect(self._on_training_error)
        self.training_worker.low_accuracy.connect(self._on_low_accuracy)
        self.training_worker.finished_ok.connect(self._on_training_finished)
        self.training_worker.finished.connect(lambda: self.btn_train_start.setEnabled(True))
        self.training_worker.start()

    def _on_training_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)

    def _on_low_accuracy(self, acc: float) -> None:
        ans = QMessageBox.question(
            self,
            "Low Accuracy",
            f"Model accuracy is {acc:.2f}% (< 60%).\nDo you want to Retry training from Step 1?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if self.training_worker:
            self.training_worker.set_retry_decision(ans == QMessageBox.StandardButton.Yes)

    def _on_training_finished(self, subject: str) -> None:
        self.big_prompt.setText(f"{subject} Model Saved")
        ans = QMessageBox.question(
            self,
            "Training Finished",
            f"{subject} model saved successfully.\n\nYes = Test Real Time classification\nNo = Go back to menu",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self.rt_subject_input.setText(subject)
            self.show_realtime()
        else:
            self.show_menu()

    def start_realtime(self) -> None:
        if self.rt_process and self.rt_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Already Running", "Real-time classifier is already running.")
            return

        subject = self.rt_subject_input.text().strip()
        if not subject:
            QMessageBox.warning(self, "Missing Subject", "Please enter subject name.")
            return

        model_path = (ROOT / f"fbcsp_lda_{subject}.joblib").resolve()
        if not model_path.exists():
            QMessageBox.critical(self, "Model Missing", f"Model file not found:\n{model_path}")
            return

        self.append_rt_log(f"\n=== Real-time session for {subject} ===")
        self.rt_status.setText("Starting real-time classification...")
        self.btn_rt_start.setEnabled(False)

        self.rt_process = QProcess(self)
        self.rt_process.setWorkingDirectory(str(ROOT))
        self.rt_process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.rt_process.readyReadStandardOutput.connect(self._on_rt_output)
        self.rt_process.finished.connect(self._on_rt_finished)

        args = [
            str(SCRIPTS / "real_time_classifier.py"),
            "--model", str(model_path),
            "--sfreq", "500",
            "--window", "4.0",
            "--step", "0.5",
            "--picks", "C3,Cz,C4",
            "--vote-k", "5",
            "--class-names", "0:hand_mi,1:rest",
        ]
        self.rt_process.start(sys.executable, args)

    def _on_rt_output(self) -> None:
        if not self.rt_process:
            return
        data = bytes(self.rt_process.readAllStandardOutput()).decode(errors="replace")
        for ln in data.splitlines():
            self.append_rt_log(ln)

    def _on_rt_finished(self, exit_code: int, _status) -> None:
        if exit_code == 0:
            self.rt_status.setText("Real-time classifier stopped.")
        else:
            self.rt_status.setText(f"Real-time classifier exited with code {exit_code}")
        self.btn_rt_start.setEnabled(True)
        self.rt_process = None

    def stop_realtime(self) -> None:
        if self.rt_process and self.rt_process.state() != QProcess.ProcessState.NotRunning:
            self.rt_status.setText("Stopping real-time classifier...")
            self.rt_process.terminate()
        else:
            self.rt_status.setText("No running real-time process.")

    def back_from_rt(self) -> None:
        self.stop_realtime()
        self.show_menu()

    def closeEvent(self, event) -> None:
        self.stop_realtime()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = EEGApp()
    win.show()
    sys.exit(app.exec())
