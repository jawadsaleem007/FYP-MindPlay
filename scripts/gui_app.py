"""
Desktop GUI for EEG FBCSP+LDA training and real-time classification.
MindPlay Edition v3.0 – Sidebar Navigation & Modern Aesthetic.

Features:
- Sidebar navigation with active-page highlighting.
- Step-progress indicators during training pipeline.
- Modern card-based layouts with gradient accents.
- IDE-style dark console with styled output.

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
from PyQt6.QtGui import QFont, QColor, QLinearGradient, QPalette, QBrush
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
    QGridLayout,
    QGraphicsDropShadowEffect,
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

                self.status.emit("Step 2/3: Training Model (FBCSP+LDA)...")
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
        self.setWindowTitle("MindPlay \u2013 EEG Control Center")
        self.resize(1320, 880)
        self.setMinimumSize(1100, 750)

        self.training_worker: Optional[TrainingWorker] = None
        self.rt_process: Optional[QProcess] = None
        self._nav_buttons: dict[str, QPushButton] = {}

        self._build_ui()
        self._apply_theme()

    def _build_ui(self) -> None:
        container = QWidget()
        self.setCentralWidget(container)
        root = QHBoxLayout(container)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Sidebar navigation
        root.addWidget(self._create_sidebar())

        # Content area
        content = QWidget()
        content.setObjectName("contentArea")
        c_lay = QVBoxLayout(content)
        c_lay.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        c_lay.addWidget(self.stack)

        self.menu_page = self._create_menu_page()
        self.train_page = self._create_training_page()
        self.rt_page = self._create_rt_page()

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.rt_page)
        root.addWidget(content, 1)
        self.show_menu()

    # ── Sidebar ─────────────────────────────────────────────

    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        lay = QVBoxLayout(sidebar)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        # Brand
        brand = QWidget()
        b_lay = QVBoxLayout(brand)
        b_lay.setContentsMargins(24, 28, 24, 24)
        b_lay.setSpacing(4)
        logo = QLabel("\u2B21 MINDPLAY")
        logo.setObjectName("sidebarTitle")
        b_lay.addWidget(logo)
        sub = QLabel("EEG CONTROL CENTER")
        sub.setObjectName("sidebarSub")
        b_lay.addWidget(sub)
        lay.addWidget(brand)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("sidebarSep")
        lay.addWidget(sep)

        # Navigation buttons
        nav = QWidget()
        n_lay = QVBoxLayout(nav)
        n_lay.setContentsMargins(0, 12, 0, 12)
        n_lay.setSpacing(2)
        for key, label in [("menu", "\u25C8   Dashboard"), ("training", "\u2699   Model Training"), ("realtime", "\u25C9   Real-Time BCI")]:
            btn = QPushButton(label)
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(46)
            self._nav_buttons[key] = btn
            n_lay.addWidget(btn)
        self._nav_buttons["menu"].clicked.connect(self.show_menu)
        self._nav_buttons["training"].clicked.connect(self.show_training)
        self._nav_buttons["realtime"].clicked.connect(self.show_realtime)
        lay.addWidget(nav)
        lay.addStretch()

        # Bottom info
        bottom = QWidget()
        bt_lay = QVBoxLayout(bottom)
        bt_lay.setContentsMargins(24, 16, 24, 20)
        bt_lay.setSpacing(4)
        self.sidebar_status = QLabel("\u25CF System Ready")
        self.sidebar_status.setObjectName("sidebarStatus")
        bt_lay.addWidget(self.sidebar_status)
        ver = QLabel("v3.0  \u00B7  Motor Imagery Pipeline")
        ver.setObjectName("sidebarVersion")
        bt_lay.addWidget(ver)
        lay.addWidget(bottom)
        return sidebar

    # ── Card Helper ─────────────────────────────────────────

    def _make_card(self, icon, icon_obj, title, desc, btn_text, btn_obj, on_click) -> QFrame:
        card = QFrame()
        card.setObjectName("menuCard")
        c_lay = QVBoxLayout(card)
        c_lay.setContentsMargins(28, 28, 28, 28)
        c_lay.setSpacing(14)
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(24)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 18))
        card.setGraphicsEffect(shadow)
        icon_lbl = QLabel(icon)
        icon_lbl.setObjectName(icon_obj)
        icon_lbl.setFixedSize(50, 50)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c_lay.addWidget(icon_lbl)
        t = QLabel(title)
        t.setObjectName("cardTitle")
        c_lay.addWidget(t)
        d = QLabel(desc)
        d.setObjectName("cardDesc")
        d.setWordWrap(True)
        c_lay.addWidget(d)
        c_lay.addStretch()
        btn = QPushButton(btn_text)
        btn.setObjectName(btn_obj)
        btn.setFixedHeight(46)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(on_click)
        c_lay.addWidget(btn)
        return card

    def _create_menu_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentArea")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 36, 40, 30)
        lay.setSpacing(28)

        # Hero banner with gradient
        hero = QFrame()
        hero.setObjectName("heroBanner")
        hero.setFixedHeight(180)
        hero_lay = QVBoxLayout(hero)
        hero_lay.setContentsMargins(44, 36, 44, 36)
        hero_lay.setSpacing(10)
        h_title = QLabel("Welcome to MindPlay")
        h_title.setObjectName("heroTitle")
        hero_lay.addWidget(h_title)
        
        h_sub = QLabel("Configure, train, and deploy your EEG motor imagery classifier.\nSelect a workflow below to get started.")
        h_sub.setObjectName("heroSub")
        hero_lay.addWidget(h_sub)
        hero_lay.addStretch()
        shadow_hero = QGraphicsDropShadowEffect()
        shadow_hero.setBlurRadius(40)
        shadow_hero.setXOffset(0)
        shadow_hero.setYOffset(12)
        shadow_hero.setColor(QColor(108, 99, 255, 50))
        hero.setGraphicsEffect(shadow_hero)
        lay.addWidget(hero)

        # Feature cards
        cards = QHBoxLayout()
        cards.setSpacing(24)
        cards.addWidget(self._make_card(
            icon="\u2699\uFE0F", icon_obj="cardIconBadge",
            title="Model Training",
            desc="Record trials via LSL, train an FBCSP+LDA\nclassifier, and evaluate performance metrics\nwith cross-validation.",
            btn_text="Launch Training  \u2192", btn_obj="cardBtn",
            on_click=self.show_training,
        ))
        cards.addWidget(self._make_card(
            icon="\U0001F9E0", icon_obj="cardIconBadgeTeal",
            title="Real-Time BCI",
            desc="Load a trained model and stream real-time\nEEG classification results for neuro-\nfeedback applications.",
            btn_text="Launch Session  \u2192", btn_obj="cardBtnTeal",
            on_click=self.show_realtime,
        ))
        lay.addLayout(cards)
        lay.addStretch()

        footer = QLabel("Data stored in /data  \u00B7  Models saved to project root")
        footer.setObjectName("footerLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(footer)
        return page

    def _create_training_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentArea")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 32, 40, 30)
        lay.setSpacing(18)

        # Page header
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        pt = QLabel("Model Training")
        pt.setObjectName("pageTitle")
        title_col.addWidget(pt)
        pd = QLabel("Record \u2192 Train \u2192 Evaluate \u2014 full automated pipeline")
        pd.setObjectName("pageDesc")
        title_col.addWidget(pd)
        header.addLayout(title_col)
        header.addStretch()
        btn_back = QPushButton("\u2190 Back")
        btn_back.setObjectName("ghost")
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self.show_menu)
        header.addWidget(btn_back)
        lay.addLayout(header)

        # Step progress indicator
        step_frame = QFrame()
        step_frame.setObjectName("stepFrame")
        step_layout = QHBoxLayout(step_frame)
        step_layout.setContentsMargins(28, 18, 28, 18)
        step_layout.setSpacing(0)
        self.step_circles: list[QLabel] = []
        self.step_labels: list[QLabel] = []
        self.step_lines: list[QFrame] = []
        for i, name in enumerate(["Record Trials", "Train Model", "Evaluate"]):
            if i > 0:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setObjectName("stepLine")
                line.setFixedHeight(2)
                step_layout.addWidget(line, 1)
                self.step_lines.append(line)
            col = QVBoxLayout()
            col.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.setSpacing(6)
            circle = QLabel(str(i + 1))
            circle.setObjectName("stepCircle")
            circle.setFixedSize(34, 34)
            circle.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(circle, alignment=Qt.AlignmentFlag.AlignCenter)
            lbl = QLabel(name)
            lbl.setObjectName("stepLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            col.addWidget(lbl)
            step_layout.addLayout(col)
            self.step_circles.append(circle)
            self.step_labels.append(lbl)
        shadow_step = QGraphicsDropShadowEffect()
        shadow_step.setBlurRadius(16)
        shadow_step.setXOffset(0)
        shadow_step.setYOffset(4)
        shadow_step.setColor(QColor(0, 0, 0, 12))
        step_frame.setGraphicsEffect(shadow_step)
        lay.addWidget(step_frame)
        self._set_training_step(0)

        # Control bar
        ctrl = QFrame()
        ctrl.setObjectName("controlCard")
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(20, 14, 20, 14)
        ctrl_lay.setSpacing(14)
        lbl = QLabel("Subject ID")
        lbl.setObjectName("inputLabel")
        ctrl_lay.addWidget(lbl)
        self.subject_input = QLineEdit("S01")
        self.subject_input.setFixedWidth(160)
        self.subject_input.setPlaceholderText("e.g. S01")
        ctrl_lay.addWidget(self.subject_input)
        self.btn_train_start = QPushButton("\u25B6  Run Full Pipeline")
        self.btn_train_start.setObjectName("primary")
        self.btn_train_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_train_start.setFixedWidth(200)
        self.btn_train_start.clicked.connect(self.start_training_pipeline)
        ctrl_lay.addWidget(self.btn_train_start)
        ctrl_lay.addStretch()
        shadow_ctrl = QGraphicsDropShadowEffect()
        shadow_ctrl.setBlurRadius(12)
        shadow_ctrl.setXOffset(0)
        shadow_ctrl.setYOffset(3)
        shadow_ctrl.setColor(QColor(0, 0, 0, 10))
        ctrl.setGraphicsEffect(shadow_ctrl)
        lay.addWidget(ctrl)

        # Status row
        status_row = QHBoxLayout()
        self.train_status = QLabel("Status: Ready")
        self.train_status.setObjectName("statusPill")
        status_row.addWidget(self.train_status)
        status_row.addStretch()
        self.big_prompt = QLabel("")
        self.big_prompt.setObjectName("bigPrompt")
        status_row.addWidget(self.big_prompt, alignment=Qt.AlignmentFlag.AlignRight)
        lay.addLayout(status_row)

        # Console
        console_card = QFrame()
        console_card.setObjectName("consoleCard")
        c_lay = QVBoxLayout(console_card)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)
        c_header = QLabel("  \u25CF  Pipeline Console Output")
        c_header.setObjectName("consoleHeader")
        c_header.setFixedHeight(36)
        c_lay.addWidget(c_header)
        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setObjectName("console")
        c_lay.addWidget(self.train_log)
        shadow_con = QGraphicsDropShadowEffect()
        shadow_con.setBlurRadius(20)
        shadow_con.setXOffset(0)
        shadow_con.setYOffset(6)
        shadow_con.setColor(QColor(0, 0, 0, 15))
        console_card.setGraphicsEffect(shadow_con)
        lay.addWidget(console_card, 1)
        return page

    def _create_rt_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentArea")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 32, 40, 30)
        lay.setSpacing(18)

        # Page header
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        pt = QLabel("Real-Time BCI")
        pt.setObjectName("pageTitle")
        title_col.addWidget(pt)
        pd = QLabel("Stream live EEG data and classify motor imagery in real time")
        pd.setObjectName("pageDesc")
        title_col.addWidget(pd)
        header.addLayout(title_col)
        header.addStretch()
        btn_back = QPushButton("\u2190 Back")
        btn_back.setObjectName("ghost")
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self.back_from_rt)
        header.addWidget(btn_back)
        lay.addLayout(header)

        # Control bar
        ctrl = QFrame()
        ctrl.setObjectName("controlCard")
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(20, 14, 20, 14)
        ctrl_lay.setSpacing(14)
        lbl = QLabel("Subject ID")
        lbl.setObjectName("inputLabel")
        ctrl_lay.addWidget(lbl)
        self.rt_subject_input = QLineEdit("S01")
        self.rt_subject_input.setFixedWidth(160)
        self.rt_subject_input.setPlaceholderText("e.g. S01")
        ctrl_lay.addWidget(self.rt_subject_input)
        self.btn_rt_start = QPushButton("\u25B6  Start Stream")
        self.btn_rt_start.setObjectName("primary")
        self.btn_rt_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rt_start.clicked.connect(self.start_realtime)
        ctrl_lay.addWidget(self.btn_rt_start)
        self.btn_rt_stop = QPushButton("\u25A0  Stop")
        self.btn_rt_stop.setObjectName("danger")
        self.btn_rt_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rt_stop.clicked.connect(self.stop_realtime)
        ctrl_lay.addWidget(self.btn_rt_stop)
        ctrl_lay.addStretch()
        shadow_ctrl = QGraphicsDropShadowEffect()
        shadow_ctrl.setBlurRadius(12)
        shadow_ctrl.setXOffset(0)
        shadow_ctrl.setYOffset(3)
        shadow_ctrl.setColor(QColor(0, 0, 0, 10))
        ctrl.setGraphicsEffect(shadow_ctrl)
        lay.addWidget(ctrl)

        self.rt_status = QLabel("Status: Idle")
        self.rt_status.setObjectName("statusPill")
        lay.addWidget(self.rt_status)

        # Console
        console_card = QFrame()
        console_card.setObjectName("consoleCard")
        c_lay = QVBoxLayout(console_card)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)
        c_header = QLabel("  \u25CF  Real-Time Neural Output")
        c_header.setObjectName("consoleHeader")
        c_header.setFixedHeight(36)
        c_lay.addWidget(c_header)
        self.rt_log = QTextEdit()
        self.rt_log.setReadOnly(True)
        self.rt_log.setObjectName("console")
        c_lay.addWidget(self.rt_log)
        shadow_con = QGraphicsDropShadowEffect()
        shadow_con.setBlurRadius(20)
        shadow_con.setXOffset(0)
        shadow_con.setYOffset(6)
        shadow_con.setColor(QColor(0, 0, 0, 15))
        console_card.setGraphicsEffect(shadow_con)
        lay.addWidget(console_card, 1)
        return page

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            /* === GLOBAL === */
            QMainWindow { background-color: #F0F2F8; font-family: 'Segoe UI', 'Roboto', sans-serif; font-size: 14px; color: #1E293B; }
            QWidget#contentArea { background-color: #F0F2F8; }

            /* === SIDEBAR === */
            QFrame#sidebar { background-color: #161A2E; border: none; }
            QFrame#sidebarSep { color: #1F2444; max-height: 1px; margin: 0 20px; }
            QLabel#sidebarTitle { color: #FFFFFF; font-size: 18px; font-weight: 800; letter-spacing: 2px; }
            QLabel#sidebarSub { color: #6C63FF; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; }
            QLabel#sidebarStatus { color: #4ECDC4; font-size: 12px; font-weight: 600; }
            QLabel#sidebarVersion { color: #4A5072; font-size: 11px; }

            QPushButton#navBtn {
                text-align: left; padding: 0 24px; border: none;
                border-left: 3px solid transparent; color: #6B7194;
                background: transparent; font-size: 14px; font-weight: 500;
                border-radius: 0;
            }
            QPushButton#navBtn:hover { background: rgba(108,99,255,0.07); color: #B8BDD6; }
            QPushButton#navBtn:checked {
                background: rgba(108,99,255,0.13); color: #FFFFFF;
                border-left: 3px solid #6C63FF; font-weight: 600;
            }

            /* === HERO BANNER === */
            QFrame#heroBanner {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #6C63FF, stop:1 #A78BFA);
                border-radius: 16px;
            }
            QLabel#heroTitle { color: #FFFFFF; font-size: 34px; font-weight: 800; }
            QLabel#heroSub { color: rgba(255,255,255,0.82); font-size: 15px; }

            /* === CARDS === */
            QFrame#menuCard { background-color: #FFFFFF; border-radius: 14px; border: 1px solid #E8ECF4; }
            QLabel#cardIconBadge {
                background-color: #EBE9FF; color: #6C63FF; font-size: 22px; border-radius: 12px;
            }
            QLabel#cardIconBadgeTeal {
                background-color: #E6FAF8; color: #4ECDC4; font-size: 22px; border-radius: 12px;
            }
            QLabel#cardTitle { font-size: 20px; font-weight: 700; color: #1E293B; }
            QLabel#cardDesc { font-size: 13px; color: #64748B; }

            QPushButton#cardBtn {
                background-color: #6C63FF; color: white; font-size: 14px;
                font-weight: 600; border-radius: 10px; border: none;
            }
            QPushButton#cardBtn:hover { background-color: #5B52E8; }
            QPushButton#cardBtnTeal {
                background-color: #4ECDC4; color: white; font-size: 14px;
                font-weight: 600; border-radius: 10px; border: none;
            }
            QPushButton#cardBtnTeal:hover { background-color: #3DB8B0; }
            QLabel#footerLabel { color: #94A3B8; font-size: 12px; }

            /* === PAGE HEADERS === */
            QLabel#pageTitle { font-size: 28px; font-weight: 800; color: #1E293B; }
            QLabel#pageDesc { font-size: 14px; color: #64748B; }

            /* === STEP INDICATOR === */
            QFrame#stepFrame { background-color: #FFFFFF; border: 1px solid #E8ECF4; border-radius: 12px; }

            /* === CONTROLS === */
            QFrame#controlCard { background-color: #FFFFFF; border: 1px solid #E8ECF4; border-radius: 12px; }
            QLabel#inputLabel { font-weight: 600; color: #475569; font-size: 13px; }

            QLineEdit {
                background-color: #F8FAFC; border: 1.5px solid #E2E8F0;
                border-radius: 8px; padding: 8px 14px; font-weight: 500; color: #1E293B;
            }
            QLineEdit:focus { border: 1.5px solid #6C63FF; background-color: #FFFFFF; }

            QPushButton#primary {
                background-color: #6C63FF; color: white; font-weight: 600;
                border-radius: 8px; padding: 10px 24px; border: none;
            }
            QPushButton#primary:hover { background-color: #5B52E8; }
            QPushButton#primary:disabled { background-color: #C4C1F7; color: white; }

            QPushButton#danger {
                background-color: #EF4444; color: white; font-weight: 600;
                border-radius: 8px; padding: 10px 24px; border: none;
            }
            QPushButton#danger:hover { background-color: #DC2626; }

            QPushButton#ghost {
                background: transparent; color: #64748B; border: 1.5px solid #E2E8F0;
                font-weight: 500; border-radius: 8px; padding: 10px 20px;
            }
            QPushButton#ghost:hover { background-color: #F8FAFC; color: #1E293B; border-color: #CBD5E1; }

            /* === STATUS === */
            QLabel#statusPill {
                background-color: #EBE9FF; color: #6C63FF; padding: 8px 18px;
                border-radius: 20px; font-weight: 700; font-size: 13px;
            }
            QLabel#bigPrompt { color: #22C55E; font-size: 20px; font-weight: 700; }

            /* === CONSOLE === */
            QFrame#consoleCard { border: 1px solid #E8ECF4; border-radius: 12px; background: #FFFFFF; }
            QLabel#consoleHeader {
                background-color: #1A1D2E; color: #6B7194;
                font-family: 'Consolas', 'JetBrains Mono', monospace;
                font-size: 12px; padding: 0 16px;
                border-top-left-radius: 11px; border-top-right-radius: 11px;
            }
            QTextEdit#console {
                background-color: #0F111A; color: #E2E8F0;
                font-family: 'Consolas', 'JetBrains Mono', 'Fira Code', monospace;
                font-size: 13px; padding: 16px; border: none;
                border-bottom-left-radius: 11px; border-bottom-right-radius: 11px;
                selection-background-color: #6C63FF; selection-color: white;
            }

            /* === SCROLLBAR === */
            QScrollBar:vertical { border: none; background: #151722; width: 10px; margin: 0; }
            QScrollBar::handle:vertical { background: #3B3F5C; min-height: 20px; border-radius: 5px; margin: 2px; }
            QScrollBar::handle:vertical:hover { background: #4F5477; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)

    # ── Navigation ──────────────────────────────────────────

    def _set_nav_active(self, key: str) -> None:
        for k, btn in self._nav_buttons.items():
            btn.setChecked(k == key)

    def show_menu(self) -> None:
        self.stack.setCurrentWidget(self.menu_page)
        self._set_nav_active("menu")

    def show_training(self) -> None:
        self.stack.setCurrentWidget(self.train_page)
        self._set_nav_active("training")

    def show_realtime(self) -> None:
        self.stack.setCurrentWidget(self.rt_page)
        self._set_nav_active("realtime")

    # ── Step Progress ───────────────────────────────────────

    def _set_training_step(self, step: int) -> None:
        """0=reset, 1=record, 2=train, 3=evaluate, 4=all done."""
        for i in range(3):
            idx = i + 1
            if step > idx:
                self.step_circles[i].setStyleSheet(
                    "background-color: #22C55E; color: white; border-radius: 17px; "
                    "font-weight: 700; font-size: 14px;"
                )
                self.step_labels[i].setStyleSheet("color: #22C55E; font-weight: 600;")
            elif step == idx:
                self.step_circles[i].setStyleSheet(
                    "background-color: #6C63FF; color: white; border-radius: 17px; "
                    "font-weight: 700; font-size: 14px;"
                )
                self.step_labels[i].setStyleSheet("color: #6C63FF; font-weight: 600;")
            else:
                self.step_circles[i].setStyleSheet(
                    "background-color: #E2E8F0; color: #94A3B8; border-radius: 17px; "
                    "font-weight: 700; font-size: 14px;"
                )
                self.step_labels[i].setStyleSheet("color: #94A3B8; font-weight: 500;")
        for i, line in enumerate(self.step_lines):
            if step > (i + 1):
                line.setStyleSheet("background-color: #22C55E;")
            else:
                line.setStyleSheet("background-color: #E2E8F0;")

    def _update_training_status(self, status_text: str) -> None:
        self.train_status.setText(f"Status: {status_text}")
        if "Step 1/3" in status_text:
            self._set_training_step(1)
        elif "Step 2/3" in status_text:
            self._set_training_step(2)
        elif "Step 3/3" in status_text:
            self._set_training_step(3)
        elif "complete" in status_text.lower():
            self._set_training_step(4)

    # ── Console Helpers ─────────────────────────────────────

    def append_train_log(self, text: str) -> None:
        self.train_log.append(text)
        sb = self.train_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_rt_log(self, text: str) -> None:
        self.rt_log.append(text)
        sb = self.rt_log.verticalScrollBar()
        sb.setValue(sb.maximum())

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
        self._set_training_step(0)
        self.train_status.setStyleSheet("")
        self.append_train_log(f"\n>>> STARTING SESSION: {subject}")

        self.training_worker = TrainingWorker(subject)
        self.training_worker.log_line.connect(self.append_train_log)
        self.training_worker.status.connect(self._update_training_status)
        self.training_worker.error.connect(self._on_training_error)
        self.training_worker.low_accuracy.connect(self._on_low_accuracy)
        self.training_worker.finished_ok.connect(self._on_training_finished)
        self.training_worker.finished.connect(lambda: self.btn_train_start.setEnabled(True))
        self.training_worker.start()

    def _on_training_error(self, title: str, message: str) -> None:
        self.train_status.setText("Status: Error")
        self.train_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        QMessageBox.critical(self, title, message)

    def _on_low_accuracy(self, acc: float) -> None:
        ans = QMessageBox.question(
            self,
            "Low Accuracy Warning",
            f"Model accuracy is {acc:.2f}% (< 60%).\nDo you want to Retry training from Step 1?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if self.training_worker:
            self.training_worker.set_retry_decision(ans == QMessageBox.StandardButton.Yes)

    def _on_training_finished(self, subject: str) -> None:
        self.big_prompt.setText("\u2713  Saved")
        self.train_status.setText("Status: Complete")
        self.train_status.setStyleSheet("background-color: #DCFCE7; color: #166534; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        self._set_training_step(4)
        self.sidebar_status.setText("\u25CF Model Ready")
        self.sidebar_status.setStyleSheet("color: #22C55E; font-size: 12px; font-weight: 600;")
        
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

        self.append_rt_log(f"\n>>> LOADING MODEL: {model_path.name}")
        self.rt_status.setText("Status: Streaming...")
        self.rt_status.setStyleSheet("background-color: #FEF3C7; color: #92400E; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
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
            self.rt_status.setText("Status: Stopped")
            self.rt_status.setStyleSheet("background-color: #EBE9FF; color: #6C63FF; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        else:
            self.rt_status.setText(f"Status: Error ({exit_code})")
            self.rt_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        self.btn_rt_start.setEnabled(True)
        self.rt_process = None

    def stop_realtime(self) -> None:
        if self.rt_process and self.rt_process.state() != QProcess.ProcessState.NotRunning:
            self.rt_status.setText("Status: Stopping...")
            self.rt_process.terminate()
        else:
            self.rt_status.setText("Status: Idle")

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