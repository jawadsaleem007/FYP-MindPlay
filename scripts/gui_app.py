"""
Desktop GUI for EEG FBCSP+LDA training and real-time classification.
MindPlay Edition v3.0 – Sidebar Navigation & Modern Aesthetic.

Features:
- Sidebar navigation with active-page highlighting.
- Step-progress indicators during training pipeline.
- Modern card-based layouts with gradient accents.
- IDE-style dark console with styled output.

Run:
    python .\\scripts\\gui_app.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import shlex
import subprocess
import queue
from pathlib import Path
from typing import Optional, Tuple, Dict

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QProcess, QTimer, QProcessEnvironment
from PyQt6.QtGui import QFont, QColor, QLinearGradient, QPalette, QBrush, QTextCursor
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
    QCheckBox,
    QScrollArea,
    QSizePolicy,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DATA_DIR = ROOT / "data"
MASTER_SCRIPT = ROOT / "start_mindplay_master.ps1"
MASTER_STATUS_FILE = ROOT / "master_launcher_status.json"
LAUNCHER_CONFIG_FILE = ROOT / "mindplay_launcher_config.json"


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
        run_cmd = list(cmd)
        if run_cmd:
            exe_name = Path(run_cmd[0]).name.lower()
            if (exe_name.startswith("python") or exe_name == "py.exe") and "-u" not in run_cmd[1:]:
                run_cmd.insert(1, "-u")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            run_cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        lines: list[str] = []
        assert proc.stdout is not None
        for raw in proc.stdout:
            # Preserve carriage-return progress updates as visible lines in GUI logs.
            normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
            parts = normalized.split("\n")
            if parts and parts[-1] == "":
                parts = parts[:-1]
            for line in parts:
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
                    "20",
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
        self.blink_process: Optional[QProcess] = None
        self.gyro_process: Optional[QProcess] = None
        self.master_process: Optional[QProcess] = None
        self.master_status_timer = QTimer(self)
        self.master_status_timer.setInterval(700)
        self.master_status_timer.timeout.connect(self._poll_master_status)
        self.master_status_file: Path = MASTER_STATUS_FILE
        self._last_master_snapshot: str = ""
        self._nav_buttons: dict[str, QPushButton] = {}
        self._module_partial_output: dict[int, str] = {}

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
        self.rt_page = self._create_mi_page()
        self.blink_page = self._create_blink_page()
        self.gyro_page = self._create_gyro_page()
        self.system_page = self._create_system_page()

        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.train_page)
        self.stack.addWidget(self.rt_page)
        self.stack.addWidget(self.blink_page)
        self.stack.addWidget(self.gyro_page)
        self.stack.addWidget(self.system_page)
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
        for key, label in [
            ("menu", "\u25C8   Dashboard"),
            ("system", "\u2692   System Launcher"),
            ("training", "\u2699   Model Training"),
            ("realtime", "\u25C9   MI Classifier"),
            ("blink", "\u25C9   Blink Detection"),
            ("gyro", "\u25C9   Gyro Detection"),
        ]:
            btn = QPushButton(label)
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setFixedHeight(46)
            self._nav_buttons[key] = btn
            n_lay.addWidget(btn)
        self._nav_buttons["menu"].clicked.connect(self.show_menu)
        self._nav_buttons["system"].clicked.connect(self.show_system)
        self._nav_buttons["training"].clicked.connect(self.show_training)
        self._nav_buttons["realtime"].clicked.connect(self.show_realtime)
        self._nav_buttons["blink"].clicked.connect(self.show_blink)
        self._nav_buttons["gyro"].clicked.connect(self.show_gyro)
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

    def _make_num_input(self, default_text: str, width: int = 120) -> QLineEdit:
        inp = QLineEdit(default_text)
        inp.setObjectName("numericInput")
        inp.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        inp.setMinimumWidth(width)
        inp.setMaximumWidth(width + 72)
        inp.setMinimumHeight(38)
        inp.setFont(QFont("Consolas", 11))
        return inp

    def _style_param_grid(self, grid: QGridLayout, columns: int) -> None:
        # Shared spacing for dense parameter forms so labels/inputs remain readable.
        grid.setContentsMargins(26, 22, 26, 22)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(16)

        if columns == 4:
            grid.setColumnStretch(1, 2)
            grid.setColumnStretch(3, 2)
        elif columns == 6:
            grid.setColumnStretch(1, 2)
            grid.setColumnStretch(3, 2)
            grid.setColumnStretch(5, 2)

    def _wrap_scroll_page(self, content: QWidget) -> QScrollArea:
        # Let page content keep natural size and scroll when viewport is smaller.
        content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        host = QWidget()
        host.setObjectName("contentArea")
        host_lay = QVBoxLayout(host)
        host_lay.setContentsMargins(0, 0, 0, 0)
        host_lay.setSpacing(0)
        host_lay.addWidget(content)
        host_lay.addStretch()

        scroll = QScrollArea()
        scroll.setObjectName("pageScroll")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(host)
        return scroll

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
        cards = QGridLayout()
        cards.setHorizontalSpacing(24)
        cards.setVerticalSpacing(24)
        card_widgets = [
            self._make_card(
                icon="\u2692\uFE0F", icon_obj="cardIconBadge",
                title="Master Launcher",
                desc="Start overlay, gyro, blink, and MI classifier\nin admin mode with one click and live\nreadiness status.",
                btn_text="Open System Launcher  \u2192", btn_obj="cardBtn",
                on_click=self.show_system,
            ),
            self._make_card(
                icon="\u2699\uFE0F", icon_obj="cardIconBadge",
                title="Model Training",
                desc="Record trials via LSL, train an FBCSP+LDA\nclassifier, and evaluate performance metrics\nwith cross-validation.",
                btn_text="Launch Training  \u2192", btn_obj="cardBtn",
                on_click=self.show_training,
            ),
            self._make_card(
                icon="\U0001F9E0", icon_obj="cardIconBadgeTeal",
                title="Real-Time BCI",
                desc="Load a trained model and stream real-time\nEEG classification results for neuro-\nfeedback applications.",
                btn_text="Launch Session  \u2192", btn_obj="cardBtnTeal",
                on_click=self.show_realtime,
            ),
            self._make_card(
                icon="\u25CE", icon_obj="cardIconBadgeSun",
                title="Blink Detection",
                desc="Detect intentional blinks from frontal EEG\nchannels and trigger optional key actions\nin real time.",
                btn_text="Open Blink Page  \u2192", btn_obj="cardBtnSun",
                on_click=self.show_blink,
            ),
            self._make_card(
                icon="\u25EC", icon_obj="cardIconBadgeSky",
                title="Gyro Detection",
                desc="Detect head movement from gyro velocity\nwith threshold, deadzone, and key\nmapping controls.",
                btn_text="Open Gyro Page  \u2192", btn_obj="cardBtnSky",
                on_click=self.show_gyro,
            ),
        ]
        for i, card in enumerate(card_widgets):
            cards.addWidget(card, i // 3, i % 3)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        cards.setColumnStretch(2, 1)
        lay.addLayout(cards)
        lay.addStretch()

        footer = QLabel("Data stored in /data  \u00B7  Models saved to project root")
        footer.setObjectName("footerLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(footer)
        return self._wrap_scroll_page(page)

    def _create_system_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentArea")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 32, 40, 30)
        lay.setSpacing(18)

        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        pt = QLabel("System Launcher")
        pt.setObjectName("pageTitle")
        title_col.addWidget(pt)
        pd = QLabel("Launch overlay + gyro + blink + real-time classifier in admin mode and monitor readiness")
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

        ctrl = QFrame()
        ctrl.setObjectName("controlCard")
        ctrl_lay = QGridLayout(ctrl)
        ctrl_lay.setContentsMargins(20, 14, 20, 14)
        ctrl_lay.setHorizontalSpacing(10)
        ctrl_lay.setVerticalSpacing(8)

        ctrl_lay.addWidget(QLabel("Model path"), 0, 0)
        self.master_model_input = QLineEdit("")
        self.master_model_input.setPlaceholderText("Optional model path (auto-picks latest fbcsp_lda*.joblib if blank)")
        ctrl_lay.addWidget(self.master_model_input, 0, 1, 1, 3)

        self.master_no_follow_cb = QCheckBox("Pin overlay to screen (disable follow-active-window)")
        ctrl_lay.addWidget(self.master_no_follow_cb, 1, 0, 1, 2)

        self.btn_master_start = QPushButton("\u25B6  Start Master (Admin)")
        self.btn_master_start.setObjectName("primary")
        self.btn_master_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_master_start.clicked.connect(self.start_master_launcher)
        ctrl_lay.addWidget(self.btn_master_start, 1, 2)

        self.btn_master_refresh = QPushButton("\u21BB  Refresh Status")
        self.btn_master_refresh.setObjectName("ghost")
        self.btn_master_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_master_refresh.clicked.connect(self._poll_master_status)
        ctrl_lay.addWidget(self.btn_master_refresh, 1, 3)

        lay.addWidget(ctrl)

        self.master_status = QLabel("Status: Idle")
        self.master_status.setObjectName("statusPill")
        lay.addWidget(self.master_status)

        comp = QFrame()
        comp.setObjectName("controlCard")
        comp_lay = QGridLayout(comp)
        comp_lay.setContentsMargins(20, 14, 20, 14)
        comp_lay.setHorizontalSpacing(14)
        comp_lay.setVerticalSpacing(8)

        comp_lay.addWidget(QLabel("Overlay"), 0, 0)
        self.master_overlay_lbl = QLabel("pending")
        self.master_overlay_lbl.setObjectName("statusPill")
        comp_lay.addWidget(self.master_overlay_lbl, 0, 1)

        comp_lay.addWidget(QLabel("Gyro"), 0, 2)
        self.master_gyro_lbl = QLabel("pending")
        self.master_gyro_lbl.setObjectName("statusPill")
        comp_lay.addWidget(self.master_gyro_lbl, 0, 3)

        comp_lay.addWidget(QLabel("Blink"), 1, 0)
        self.master_blink_lbl = QLabel("pending")
        self.master_blink_lbl.setObjectName("statusPill")
        comp_lay.addWidget(self.master_blink_lbl, 1, 1)

        comp_lay.addWidget(QLabel("Classifier"), 1, 2)
        self.master_classifier_lbl = QLabel("pending")
        self.master_classifier_lbl.setObjectName("statusPill")
        comp_lay.addWidget(self.master_classifier_lbl, 1, 3)

        self._master_component_widgets: Dict[str, QLabel] = {
            "overlay": self.master_overlay_lbl,
            "gyro": self.master_gyro_lbl,
            "blink": self.master_blink_lbl,
            "classifier": self.master_classifier_lbl,
        }
        lay.addWidget(comp)

        console_card = QFrame()
        console_card.setObjectName("consoleCard")
        c_lay = QVBoxLayout(console_card)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)
        c_header = QLabel("  \u25CF  Master Launcher Status")
        c_header.setObjectName("consoleHeader")
        c_header.setFixedHeight(36)
        c_lay.addWidget(c_header)
        self.master_log = QTextEdit()
        self.master_log.setReadOnly(True)
        self.master_log.setObjectName("console")
        c_lay.addWidget(self.master_log)
        console_card.setMinimumHeight(280)
        lay.addWidget(console_card)

        return self._wrap_scroll_page(page)

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
        console_card.setMinimumHeight(300)
        lay.addWidget(console_card)
        return self._wrap_scroll_page(page)

    def _create_mi_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentArea")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 32, 40, 30)
        lay.setSpacing(18)

        # Page header
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        pt = QLabel("MI Classifier")
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
        self.btn_rt_start = QPushButton("\u25B6  Start MI")
        self.btn_rt_start.setObjectName("primary")
        self.btn_rt_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_rt_start.clicked.connect(self.start_realtime)
        ctrl_lay.addWidget(self.btn_rt_start)
        self.btn_rt_stop = QPushButton("\u25A0  Stop MI")
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

        cfg = QFrame()
        cfg.setObjectName("controlCard")
        cfg_lay = QGridLayout(cfg)
        self._style_param_grid(cfg_lay, columns=4)
        cfg_lay.addWidget(QLabel("Model path"), 0, 0)
        self.mi_model_input = QLineEdit("")
        self.mi_model_input.setPlaceholderText("Optional custom model path (default: fbcsp_lda_<subject>.joblib)")
        cfg_lay.addWidget(self.mi_model_input, 0, 1, 1, 3)
        cfg_lay.addWidget(QLabel("MI sfreq"), 1, 0)
        self.mi_sfreq_input = self._make_num_input("500")
        cfg_lay.addWidget(self.mi_sfreq_input, 1, 1)
        cfg_lay.addWidget(QLabel("Window (s)"), 1, 2)
        self.mi_window_input = self._make_num_input("4.0")
        cfg_lay.addWidget(self.mi_window_input, 1, 3)
        cfg_lay.addWidget(QLabel("Step (s)"), 2, 0)
        self.mi_step_input = self._make_num_input("0.5")
        cfg_lay.addWidget(self.mi_step_input, 2, 1)
        cfg_lay.addWidget(QLabel("Vote-k"), 2, 2)
        self.mi_vote_input = self._make_num_input("5")
        cfg_lay.addWidget(self.mi_vote_input, 2, 3)
        cfg_lay.addWidget(QLabel("Picks"), 3, 0)
        self.mi_picks_input = QLineEdit("C3,Cz,C4")
        cfg_lay.addWidget(self.mi_picks_input, 3, 1)
        cfg_lay.addWidget(QLabel("Class names"), 3, 2)
        self.mi_classes_input = QLineEdit("0:rest,1:hand_mi")
        cfg_lay.addWidget(self.mi_classes_input, 3, 3)
        cfg_lay.addWidget(QLabel("Hand MI threshold"), 4, 0)
        self.mi_hand_thr_input = self._make_num_input("0.9")
        cfg_lay.addWidget(self.mi_hand_thr_input, 4, 1)
        cfg_lay.addWidget(QLabel("Consecutive windows"), 4, 2)
        self.mi_hand_consec_input = self._make_num_input("2")
        cfg_lay.addWidget(self.mi_hand_consec_input, 4, 3)
        self.mi_scale_uv_cb = QCheckBox("Scale incoming values to uV")
        cfg_lay.addWidget(self.mi_scale_uv_cb, 5, 0, 1, 2)
        self.mi_block_cb = QCheckBox("Use non-overlapping windows (block mode)")
        cfg_lay.addWidget(self.mi_block_cb, 5, 2, 1, 2)
        lay.addWidget(cfg)

        self.mi_status = QLabel("Status: Idle")
        self.mi_status.setObjectName("statusPill")
        lay.addWidget(self.mi_status)

        # Console
        console_card = QFrame()
        console_card.setObjectName("consoleCard")
        c_lay = QVBoxLayout(console_card)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)
        c_header = QLabel("  \u25CF  MI Runtime Output")
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
        console_card.setMinimumHeight(280)
        lay.addWidget(console_card)
        return self._wrap_scroll_page(page)

    def _create_blink_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentArea")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 32, 40, 30)
        lay.setSpacing(18)

        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        pt = QLabel("Blink Detection")
        pt.setObjectName("pageTitle")
        title_col.addWidget(pt)
        pd = QLabel("Run frontal-channel blink detection and optional key actions")
        pd.setObjectName("pageDesc")
        title_col.addWidget(pd)
        header.addLayout(title_col)
        header.addStretch()
        btn_back = QPushButton("\u2190 Back")
        btn_back.setObjectName("ghost")
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self.back_from_blink)
        header.addWidget(btn_back)
        lay.addLayout(header)

        ctrl = QFrame()
        ctrl.setObjectName("controlCard")
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(20, 14, 20, 14)
        ctrl_lay.setSpacing(14)
        self.btn_blink_start = QPushButton("\u25B6  Start Blink")
        self.btn_blink_start.setObjectName("primary")
        self.btn_blink_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_blink_start.clicked.connect(self.start_blink)
        ctrl_lay.addWidget(self.btn_blink_start)
        self.btn_blink_stop = QPushButton("\u25A0  Stop Blink")
        self.btn_blink_stop.setObjectName("danger")
        self.btn_blink_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_blink_stop.clicked.connect(self.stop_blink)
        ctrl_lay.addWidget(self.btn_blink_stop)
        ctrl_lay.addStretch()
        lay.addWidget(ctrl)

        cfg = QFrame()
        cfg.setObjectName("controlCard")
        cfg_lay = QGridLayout(cfg)
        self._style_param_grid(cfg_lay, columns=4)
        cfg_lay.addWidget(QLabel("Blink sfreq"), 0, 0)
        self.blink_sfreq_input = QLineEdit("500")
        cfg_lay.addWidget(self.blink_sfreq_input, 0, 1)
        cfg_lay.addWidget(QLabel("Picks"), 0, 2)
        self.blink_picks_input = QLineEdit("Fp1,Fp2")
        cfg_lay.addWidget(self.blink_picks_input, 0, 3)
        cfg_lay.addWidget(QLabel("Window (s)"), 1, 0)
        self.blink_window_input = QLineEdit("0.5")
        cfg_lay.addWidget(self.blink_window_input, 1, 1)
        cfg_lay.addWidget(QLabel("Threshold (uV)"), 1, 2)
        self.blink_thr_input = QLineEdit("140")
        cfg_lay.addWidget(self.blink_thr_input, 1, 3)
        cfg_lay.addWidget(QLabel("Refractory (s)"), 2, 0)
        self.blink_refractory_input = QLineEdit("0.8")
        cfg_lay.addWidget(self.blink_refractory_input, 2, 1)
        cfg_lay.addWidget(QLabel("Key (optional)"), 2, 2)
        self.blink_key_input = QLineEdit("enter")
        cfg_lay.addWidget(self.blink_key_input, 2, 3)
        self.blink_scale_uv_cb = QCheckBox("Scale incoming values to uV")
        self.blink_scale_uv_cb.setChecked(True)
        cfg_lay.addWidget(self.blink_scale_uv_cb, 3, 0, 1, 2)
        self.blink_extra_input = QLineEdit("")
        self.blink_extra_input.setPlaceholderText("Extra blink args (optional)")
        cfg_lay.addWidget(self.blink_extra_input, 3, 2, 1, 2)
        lay.addWidget(cfg)

        self.blink_status = QLabel("Status: Idle")
        self.blink_status.setObjectName("statusPill")
        lay.addWidget(self.blink_status)

        console_card = QFrame()
        console_card.setObjectName("consoleCard")
        c_lay = QVBoxLayout(console_card)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)
        c_header = QLabel("  \u25CF  Blink Detector Output")
        c_header.setObjectName("consoleHeader")
        c_header.setFixedHeight(36)
        c_lay.addWidget(c_header)
        self.blink_log = QTextEdit()
        self.blink_log.setReadOnly(True)
        self.blink_log.setObjectName("console")
        c_lay.addWidget(self.blink_log)
        console_card.setMinimumHeight(260)
        lay.addWidget(console_card)
        return self._wrap_scroll_page(page)

    def _create_gyro_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("contentArea")
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 32, 40, 30)
        lay.setSpacing(18)

        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        pt = QLabel("Gyro Detection")
        pt.setObjectName("pageTitle")
        title_col.addWidget(pt)
        pd = QLabel("Run velocity-based gyroscope detection with configurable thresholds")
        pd.setObjectName("pageDesc")
        title_col.addWidget(pd)
        header.addLayout(title_col)
        header.addStretch()
        btn_back = QPushButton("\u2190 Back")
        btn_back.setObjectName("ghost")
        btn_back.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_back.clicked.connect(self.back_from_gyro)
        header.addWidget(btn_back)
        lay.addLayout(header)

        ctrl = QFrame()
        ctrl.setObjectName("controlCard")
        ctrl_lay = QHBoxLayout(ctrl)
        ctrl_lay.setContentsMargins(20, 14, 20, 14)
        ctrl_lay.setSpacing(14)
        self.btn_gyro_start = QPushButton("\u25B6  Start Gyro")
        self.btn_gyro_start.setObjectName("primary")
        self.btn_gyro_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_gyro_start.clicked.connect(self.start_gyro)
        ctrl_lay.addWidget(self.btn_gyro_start)
        self.btn_gyro_stop = QPushButton("\u25A0  Stop Gyro")
        self.btn_gyro_stop.setObjectName("danger")
        self.btn_gyro_stop.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_gyro_stop.clicked.connect(self.stop_gyro)
        ctrl_lay.addWidget(self.btn_gyro_stop)
        ctrl_lay.addStretch()
        lay.addWidget(ctrl)

        cfg = QFrame()
        cfg.setObjectName("controlCard")
        cfg_lay = QGridLayout(cfg)
        self._style_param_grid(cfg_lay, columns=4)
        cfg_lay.addWidget(QLabel("Gyro sfreq"), 0, 0)
        self.gyro_sfreq_input = self._make_num_input("500")
        cfg_lay.addWidget(self.gyro_sfreq_input, 0, 1)
        cfg_lay.addWidget(QLabel("Channels"), 0, 2)
        self.gyro_channels_input = QLineEdit("5,6,7")
        cfg_lay.addWidget(self.gyro_channels_input, 0, 3)
        cfg_lay.addWidget(QLabel("Stream type"), 1, 0)
        self.gyro_stream_type_input = QLineEdit("EEG")
        cfg_lay.addWidget(self.gyro_stream_type_input, 1, 1)
        cfg_lay.addWidget(QLabel("Scale factor"), 1, 2)
        self.gyro_scale_input = self._make_num_input("0.25")
        cfg_lay.addWidget(self.gyro_scale_input, 1, 3)
        cfg_lay.addWidget(QLabel("Vel Forward"), 2, 0)
        self.gyro_vel_f_input = self._make_num_input("30")
        cfg_lay.addWidget(self.gyro_vel_f_input, 2, 1)
        cfg_lay.addWidget(QLabel("Vel Backward"), 2, 2)
        self.gyro_vel_b_input = self._make_num_input("30")
        cfg_lay.addWidget(self.gyro_vel_b_input, 2, 3)
        cfg_lay.addWidget(QLabel("Vel Left"), 3, 0)
        self.gyro_vel_l_input = self._make_num_input("80")
        cfg_lay.addWidget(self.gyro_vel_l_input, 3, 1)
        cfg_lay.addWidget(QLabel("Vel Right"), 3, 2)
        self.gyro_vel_r_input = self._make_num_input("80")
        cfg_lay.addWidget(self.gyro_vel_r_input, 3, 3)
        cfg_lay.addWidget(QLabel("Vel Return"), 4, 0)
        self.gyro_vel_return_input = self._make_num_input("120")
        cfg_lay.addWidget(self.gyro_vel_return_input, 4, 1)

        cfg_lay.addWidget(QLabel("Deadzone X"), 5, 0)
        self.gyro_deadzone_x_input = self._make_num_input("5")
        cfg_lay.addWidget(self.gyro_deadzone_x_input, 5, 1)
        cfg_lay.addWidget(QLabel("Deadzone Y"), 5, 2)
        self.gyro_deadzone_y_input = self._make_num_input("15")
        cfg_lay.addWidget(self.gyro_deadzone_y_input, 5, 3)
        cfg_lay.addWidget(QLabel("Deadzone Z"), 6, 0)
        self.gyro_deadzone_z_input = self._make_num_input("15")
        cfg_lay.addWidget(self.gyro_deadzone_z_input, 6, 1)

        cfg_lay.addWidget(QLabel("Calibration (s)"), 6, 2)
        self.gyro_calib_input = self._make_num_input("2.0")
        cfg_lay.addWidget(self.gyro_calib_input, 6, 3)
        cfg_lay.addWidget(QLabel("Smoothing"), 7, 0)
        self.gyro_smoothing_input = self._make_num_input("14")
        cfg_lay.addWidget(self.gyro_smoothing_input, 7, 1)
        cfg_lay.addWidget(QLabel("Gamepad repeat"), 7, 2)
        self.gyro_repeat_input = self._make_num_input("0.40")
        cfg_lay.addWidget(self.gyro_repeat_input, 7, 3)

        self.gyro_use_z_lr_cb = QCheckBox("Use Z axis for Left/Right")
        self.gyro_use_z_lr_cb.setChecked(True)
        cfg_lay.addWidget(self.gyro_use_z_lr_cb, 8, 0, 1, 2)
        cfg_lay.addWidget(QLabel("Z Left threshold"), 8, 2)
        self.gyro_z_left_input = self._make_num_input("20")
        cfg_lay.addWidget(self.gyro_z_left_input, 8, 3)
        cfg_lay.addWidget(QLabel("Z Right threshold"), 9, 0)
        self.gyro_z_right_input = self._make_num_input("20")
        cfg_lay.addWidget(self.gyro_z_right_input, 9, 1)

        self.gyro_gamepad_mode_cb = QCheckBox("Gamepad mode")
        self.gyro_gamepad_mode_cb.setChecked(True)
        cfg_lay.addWidget(self.gyro_gamepad_mode_cb, 9, 2)
        self.gyro_output_keys_cb = QCheckBox("Output keypresses")
        self.gyro_output_keys_cb.setChecked(True)
        cfg_lay.addWidget(self.gyro_output_keys_cb, 9, 3)
        self.gyro_verbose_cb = QCheckBox("Verbose")
        self.gyro_verbose_cb.setChecked(True)
        cfg_lay.addWidget(self.gyro_verbose_cb, 10, 0)

        self.gyro_drift_cb = QCheckBox("Enable drift correction")
        cfg_lay.addWidget(self.gyro_drift_cb, 10, 1, 1, 2)

        cfg_lay.addWidget(QLabel("Overlay state file"), 11, 0)
        self.gyro_overlay_state_input = QLineEdit(str((ROOT / "gamepad_state.json").resolve()))
        cfg_lay.addWidget(self.gyro_overlay_state_input, 11, 1, 1, 3)

        cfg_lay.addWidget(QLabel("Key mapping"), 12, 0)
        self.gyro_keymap_input = QLineEdit("forward:w,backward:s,left:a,right:d")
        cfg_lay.addWidget(self.gyro_keymap_input, 12, 1, 1, 3)
        self.gyro_invert_x_cb = QCheckBox("Invert X")
        self.gyro_invert_y_cb = QCheckBox("Invert Y")
        self.gyro_invert_z_cb = QCheckBox("Invert Z")
        cfg_lay.addWidget(self.gyro_invert_x_cb, 13, 0)
        cfg_lay.addWidget(self.gyro_invert_y_cb, 13, 1)
        cfg_lay.addWidget(self.gyro_invert_z_cb, 13, 2)
        cfg_lay.addWidget(QLabel("Extra args"), 14, 0)
        self.gyro_extra_input = QLineEdit("")
        self.gyro_extra_input.setPlaceholderText("Extra gyro args (optional)")
        cfg_lay.addWidget(self.gyro_extra_input, 14, 1, 1, 3)
        lay.addWidget(cfg)

        self.gyro_status = QLabel("Status: Idle")
        self.gyro_status.setObjectName("statusPill")
        lay.addWidget(self.gyro_status)

        console_card = QFrame()
        console_card.setObjectName("consoleCard")
        c_lay = QVBoxLayout(console_card)
        c_lay.setContentsMargins(0, 0, 0, 0)
        c_lay.setSpacing(0)
        c_header = QLabel("  \u25CF  Gyro Detector Output")
        c_header.setObjectName("consoleHeader")
        c_header.setFixedHeight(36)
        c_lay.addWidget(c_header)
        self.gyro_log = QTextEdit()
        self.gyro_log.setReadOnly(True)
        self.gyro_log.setObjectName("console")
        c_lay.addWidget(self.gyro_log)
        console_card.setMinimumHeight(280)
        lay.addWidget(console_card)
        return self._wrap_scroll_page(page)

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
            QLabel#cardIconBadgeSun {
                background-color: #FFF4D6; color: #F59E0B; font-size: 22px; border-radius: 12px;
            }
            QLabel#cardIconBadgeSky {
                background-color: #E0F2FE; color: #0284C7; font-size: 22px; border-radius: 12px;
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
            QPushButton#cardBtnSun {
                background-color: #F59E0B; color: white; font-size: 14px;
                font-weight: 600; border-radius: 10px; border: none;
            }
            QPushButton#cardBtnSun:hover { background-color: #D97706; }
            QPushButton#cardBtnSky {
                background-color: #0284C7; color: white; font-size: 14px;
                font-weight: 600; border-radius: 10px; border: none;
            }
            QPushButton#cardBtnSky:hover { background-color: #0369A1; }
            QLabel#footerLabel { color: #94A3B8; font-size: 12px; }

            /* === PAGE HEADERS === */
            QLabel#pageTitle { font-size: 28px; font-weight: 800; color: #1E293B; }
            QLabel#pageDesc { font-size: 14px; color: #64748B; }

            /* === STEP INDICATOR === */
            QFrame#stepFrame { background-color: #FFFFFF; border: 1px solid #E8ECF4; border-radius: 12px; }

            /* === CONTROLS === */
            QFrame#controlCard { background-color: #FFFFFF; border: 1px solid #E8ECF4; border-radius: 12px; }
            QLabel#inputLabel { font-weight: 600; color: #475569; font-size: 13px; }
            QFrame#controlCard QLabel { color: #475569; font-size: 13px; font-weight: 600; }
            QFrame#controlCard QCheckBox { color: #475569; font-weight: 500; }

            QLineEdit {
                background-color: #F8FAFC; border: 1.5px solid #E2E8F0;
                border-radius: 8px; padding: 8px 14px; font-weight: 500; color: #1E293B;
                min-height: 38px; font-size: 14px;
            }
            QLineEdit:focus { border: 1.5px solid #6C63FF; background-color: #FFFFFF; }

            QLineEdit#numericInput {
                font-family: 'Consolas', 'JetBrains Mono', 'Fira Code', monospace;
                font-size: 14px;
                font-weight: 600;
                letter-spacing: 0px;
                min-height: 38px;
                padding: 8px 10px;
            }

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

            /* === PAGE SCROLLBARS === */
            QScrollArea#pageScroll { border: none; background-color: #F0F2F8; }
            QScrollArea#pageScroll QWidget#contentArea { background-color: #F0F2F8; }
            QScrollArea#pageScroll QScrollBar:vertical { border: none; background: #E2E8F0; width: 12px; margin: 0; }
            QScrollArea#pageScroll QScrollBar::handle:vertical { background: #94A3B8; min-height: 32px; border-radius: 6px; margin: 2px; }
            QScrollArea#pageScroll QScrollBar::handle:vertical:hover { background: #64748B; }
            QScrollArea#pageScroll QScrollBar::add-line:vertical, QScrollArea#pageScroll QScrollBar::sub-line:vertical { height: 0; }
            QScrollArea#pageScroll QScrollBar::add-page:vertical, QScrollArea#pageScroll QScrollBar::sub-page:vertical { background: none; }

            /* === CONSOLE SCROLLBARS === */
            QTextEdit#console QScrollBar:vertical { border: none; background: #151722; width: 10px; margin: 0; }
            QTextEdit#console QScrollBar::handle:vertical { background: #3B3F5C; min-height: 20px; border-radius: 5px; margin: 2px; }
            QTextEdit#console QScrollBar::handle:vertical:hover { background: #4F5477; }
            QTextEdit#console QScrollBar::add-line:vertical, QTextEdit#console QScrollBar::sub-line:vertical { height: 0; }
            QTextEdit#console QScrollBar::add-page:vertical, QTextEdit#console QScrollBar::sub-page:vertical { background: none; }
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

    def show_system(self) -> None:
        self.stack.setCurrentWidget(self.system_page)
        self._set_nav_active("system")
        self._poll_master_status()

    def show_realtime(self) -> None:
        self.stack.setCurrentWidget(self.rt_page)
        self._set_nav_active("realtime")

    def show_blink(self) -> None:
        self.stack.setCurrentWidget(self.blink_page)
        self._set_nav_active("blink")

    def show_gyro(self) -> None:
        self.stack.setCurrentWidget(self.gyro_page)
        self._set_nav_active("gyro")

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

    @staticmethod
    def _append_console_line(widget: QTextEdit, text: str) -> None:
        widget.moveCursor(QTextCursor.MoveOperation.End)
        widget.insertPlainText(f"{text}\n")
        sb = widget.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_train_log(self, text: str) -> None:
        self._append_console_line(self.train_log, text)

    def append_rt_log(self, text: str) -> None:
        self._append_console_line(self.rt_log, text)

    def append_blink_log(self, text: str) -> None:
        self._append_console_line(self.blink_log, text)

    def append_gyro_log(self, text: str) -> None:
        self._append_console_line(self.gyro_log, text)

    def append_master_log(self, text: str) -> None:
        self._append_console_line(self.master_log, text)

    def _set_pill_style(self, label: QLabel, state: str, extra: str = "") -> None:
        s = state.lower().strip()
        if any(k in s for k in ("running", "ready", "launched")):
            style = "background-color: #DCFCE7; color: #166534; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;"
        elif any(k in s for k in ("starting", "elevating", "checking", "pending", "installing")):
            style = "background-color: #FEF3C7; color: #92400E; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;"
        elif any(k in s for k in ("error", "failed")):
            style = "background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;"
        else:
            style = "background-color: #EBE9FF; color: #6C63FF; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;"
        label.setStyleSheet(style)
        label.setText(f"{state}{extra}")

    def _find_latest_model(self) -> Optional[Path]:
        candidates = sorted(
            ROOT.glob("fbcsp_lda*.joblib"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _resolve_master_model_path(self) -> Optional[Path]:
        model_input = self.master_model_input.text().strip()
        if model_input:
            requested = Path(model_input)
            candidates: list[Path] = []
            if requested.is_absolute():
                candidates.append(requested)
            else:
                candidates.append((ROOT / requested).resolve())
                if requested.parts and requested.parts[0].lower() == ROOT.name.lower() and len(requested.parts) > 1:
                    trimmed = Path(*requested.parts[1:])
                    candidates.append((ROOT / trimmed).resolve())
                candidates.append((ROOT / requested.name).resolve())

            for candidate in candidates:
                if candidate.exists():
                    self.master_model_input.setText(str(candidate))
                    return candidate

            candidate = candidates[0]

            requested_name = Path(model_input).name.lower()
            if requested_name == "fbcsp_lda.joblib":
                latest = self._find_latest_model()
                if latest:
                    self.master_model_input.setText(str(latest))
                    return latest

            QMessageBox.critical(
                self,
                "Model Missing",
                f"Model file not found:\n{candidate}\n\n"
                "Use an existing .joblib path or train a model first.",
            )
            return None

        latest = self._find_latest_model()
        if latest:
            self.master_model_input.setText(str(latest))
            return latest

        QMessageBox.critical(
            self,
            "No Model Found",
            "No model file was found in project root (expected pattern: fbcsp_lda*.joblib).\n"
            "Run Model Training first, then retry System Launcher.",
        )
        return None

    def _save_launcher_config(self) -> bool:
        """Save current dashboard parameters to launcher config file."""
        try:
            config = {
                "blink": {
                    "sfreq": self.blink_sfreq_input.text().strip() or "500",
                    "picks": self.blink_picks_input.text().strip() or "Fp1,Fp2",
                    "window": self.blink_window_input.text().strip() or "0.5",
                    "threshold_uv": self.blink_thr_input.text().strip() or "140",
                    "refractory": self.blink_refractory_input.text().strip() or "0.8",
                    "key": self.blink_key_input.text().strip() or "enter",
                    "scale_to_uv": self.blink_scale_uv_cb.isChecked(),
                    "extra_args": self.blink_extra_input.text().strip() or "",
                },
                "classifier": {
                    "sfreq": self.mi_sfreq_input.text().strip() or "500",
                    "window": self.mi_window_input.text().strip() or "4.0",
                    "step": self.mi_step_input.text().strip() or "0.5",
                    "picks": self.mi_picks_input.text().strip() or "C3,Cz,C4",
                    "vote_k": self.mi_vote_input.text().strip() or "5",
                    "class_names": self.mi_classes_input.text().strip() or "0:rest,1:hand_mi",
                    "hand_mi_threshold": self.mi_hand_thr_input.text().strip() or "0.9",
                    "hand_mi_consecutive": self.mi_hand_consec_input.text().strip() or "2",
                    "scale_to_uV": self.mi_scale_uv_cb.isChecked(),
                    "block": self.mi_block_cb.isChecked(),
                }
            }
            with open(LAUNCHER_CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            self.append_master_log(f">>> [ERROR] Failed to save launcher config: {e}")
            return False

    def start_master_launcher(self) -> None:
        if not MASTER_SCRIPT.exists():
            QMessageBox.critical(self, "Master Script Missing", f"Could not find:\n{MASTER_SCRIPT}")
            return

        if self.master_process and self.master_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Already Running", "Master launcher process is already running.")
            return

        # Save dashboard parameters to config file
        if not self._save_launcher_config():
            QMessageBox.warning(self, "Config Save Failed", "Failed to save launcher configuration.")
            return

        resolved_model = self._resolve_master_model_path()
        if not resolved_model:
            return
        model_arg = str(resolved_model)

        self.master_status_file = MASTER_STATUS_FILE
        try:
            if self.master_status_file.exists():
                self.master_status_file.unlink()
        except Exception:
            pass

        self._last_master_snapshot = ""
        self.master_log.clear()
        self.append_master_log(f">>> [MASTER] Starting launcher script: {MASTER_SCRIPT}")
        self.append_master_log(f">>> [MASTER] Status file: {self.master_status_file}")
        self.append_master_log(f">>> [MASTER] Config file: {LAUNCHER_CONFIG_FILE}")
        self.append_master_log(f">>> [MASTER] Using model: {model_arg}")

        self._set_pill_style(self.master_status, "starting", " (requesting admin)")
        for lbl in self._master_component_widgets.values():
            self._set_pill_style(lbl, "pending")

        args = [
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(MASTER_SCRIPT),
            "-StatusFile",
            str(self.master_status_file),
            "-ConfigFile",
            str(LAUNCHER_CONFIG_FILE),
        ]
        if model_arg:
            args.extend(["-ModelPath", model_arg])
        if self.master_no_follow_cb.isChecked():
            args.append("-NoOverlayFollow")

        proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_master_output)
        proc.finished.connect(self._on_master_finished)
        proc.start("powershell.exe", args)
        if not proc.waitForStarted(4000):
            self._set_pill_style(self.master_status, "error", " (failed to launch)")
            self.append_master_log(">>> [MASTER] Failed to start powershell launcher process")
            return

        self.master_process = proc
        self.btn_master_start.setEnabled(False)
        self.master_status_timer.start()
        self.sidebar_status.setText("\u25CF Master Launch In Progress")
        self.sidebar_status.setStyleSheet("color: #F59E0B; font-size: 12px; font-weight: 600;")

    def _on_master_output(self) -> None:
        if not self.master_process:
            return
        data = bytes(self.master_process.readAllStandardOutput()).decode(errors="replace")
        for ln in data.splitlines():
            self.append_master_log(f"[MASTER] {ln}")

    def _on_master_finished(self, exit_code: int, _status) -> None:
        self.append_master_log(f">>> [MASTER] Launcher process exited with code {exit_code}")
        self.master_process = None
        # Keep polling status file because elevated process may still be running.
        self.btn_master_start.setEnabled(True)

    def _poll_master_status(self) -> None:
        if not self.master_status_file.exists():
            return

        try:
            raw = self.master_status_file.read_text(encoding="utf-8-sig").lstrip("\ufeff").strip()
            if not raw or raw == self._last_master_snapshot:
                return
            self._last_master_snapshot = raw
            payload = json.loads(raw)
        except Exception as exc:
            self.append_master_log(f">>> [MASTER] Status parse error: {exc}")
            return

        phase = str(payload.get("phase", "unknown"))
        msg = str(payload.get("message", "")).strip()
        phase_text = phase if not msg else f"{phase} - {msg}"
        self._set_pill_style(self.master_status, phase, f" - {msg}" if msg else "")
        self.append_master_log(f">>> [MASTER] {phase_text}")

        components = payload.get("components", {})
        if isinstance(components, dict):
            for name, widget in self._master_component_widgets.items():
                item = components.get(name, {}) if isinstance(components.get(name, {}), dict) else {}
                c_state = str(item.get("state", "pending"))
                pid = int(item.get("pid", 0) or 0)
                extra = f" (pid {pid})" if pid > 0 else ""
                self._set_pill_style(widget, c_state, extra)

        if phase.lower() == "ready":
            self.sidebar_status.setText("\u25CF System Ready")
            self.sidebar_status.setStyleSheet("color: #22C55E; font-size: 12px; font-weight: 600;")
        elif phase.lower() == "error":
            self.sidebar_status.setText("\u25CF System Error")
            self.sidebar_status.setStyleSheet("color: #EF4444; font-size: 12px; font-weight: 600;")

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
            QMessageBox.information(self, "Already Running", "MI classifier is already running.")
            return

        model_override = self.mi_model_input.text().strip()
        if model_override:
            model_path = Path(model_override)
            if not model_path.is_absolute():
                model_path = (ROOT / model_path).resolve()
        else:
            subject = self.rt_subject_input.text().strip()
            if not subject:
                QMessageBox.warning(self, "Missing Subject", "Please enter subject name or provide a model path.")
                return
            model_path = (ROOT / f"fbcsp_lda_{subject}.joblib").resolve()

        if not model_path.exists():
            QMessageBox.critical(self, "Model Missing", f"Model file not found:\n{model_path}")
            return

        self.append_rt_log(f"\n>>> [MI] LOADING MODEL: {model_path.name}")
        self.rt_status.setText("Status: Starting...")
        self.rt_status.setStyleSheet("background-color: #FEF3C7; color: #92400E; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        self.btn_rt_start.setEnabled(False)

        mi_args = [
            str(SCRIPTS / "real_time_classifier.py"),
            "--model", str(model_path),
            "--sfreq", self.mi_sfreq_input.text().strip() or "500",
            "--window", self.mi_window_input.text().strip() or "4.0",
            "--step", self.mi_step_input.text().strip() or "0.5",
            "--picks", self.mi_picks_input.text().strip() or "C3,Cz,C4",
            "--vote-k", self.mi_vote_input.text().strip() or "5",
            "--class-names", self.mi_classes_input.text().strip() or "0:rest,1:hand_mi",
            "--hand-mi-threshold", self.mi_hand_thr_input.text().strip() or "0.97",
            "--hand-mi-consecutive", self.mi_hand_consec_input.text().strip() or "3",
        ]
        if self.mi_scale_uv_cb.isChecked():
            mi_args.append("--scale-to-uv")
        if self.mi_block_cb.isChecked():
            mi_args.append("--block")
        self.rt_process = self._start_module_process("MI", mi_args, self.append_rt_log, self._on_rt_finished)
        if self.rt_process:
            self.mi_status.setText("Status: Running")
            self.mi_status.setStyleSheet("background-color: #DCFCE7; color: #166534; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
            self.rt_status.setText("Status: Running")
            self.rt_status.setStyleSheet("background-color: #DCFCE7; color: #166534; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        else:
            self.rt_status.setText("Status: Error")
            self.rt_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
            self.btn_rt_start.setEnabled(True)

    def _start_module_process(self, module_name: str, args: list[str], append_fn, finished_fn) -> Optional[QProcess]:
        proc = QProcess(self)
        proc.setWorkingDirectory(str(ROOT))
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)

        run_args = list(args)
        if "-u" not in run_args:
            run_args.insert(0, "-u")

        self._module_partial_output[id(proc)] = ""
        proc.readyReadStandardOutput.connect(lambda n=module_name, p=proc, fn=append_fn: self._on_module_output(n, p, fn))
        proc.finished.connect(
            lambda exit_code, _status, n=module_name, p=proc, afn=append_fn, fn=finished_fn:
            self._on_module_finished(n, p, afn, fn, exit_code)
        )
        proc.start(sys.executable, run_args)
        if not proc.waitForStarted(3000):
            self._module_partial_output.pop(id(proc), None)
            append_fn(f">>> [{module_name}] Failed to start process")
            return None
        return proc

    def _on_module_output(self, module_name: str, proc: QProcess, append_fn) -> None:
        data = bytes(proc.readAllStandardOutput()).decode(errors="replace")
        if not data:
            return

        key = id(proc)
        pending = self._module_partial_output.get(key, "")
        normalized = (pending + data).replace("\r\n", "\n").replace("\r", "\n")
        parts = normalized.split("\n")
        self._module_partial_output[key] = parts.pop() if parts else ""

        for ln in parts:
            append_fn(f"[{module_name}] {ln}")

    def _on_module_finished(self, module_name: str, proc: QProcess, append_fn, finished_fn, exit_code: int) -> None:
        self._on_module_output(module_name, proc, append_fn)
        tail = self._module_partial_output.pop(id(proc), "")
        if tail:
            append_fn(f"[{module_name}] {tail}")
        finished_fn(exit_code)

    def _on_rt_finished(self, exit_code: int) -> None:
        self.rt_process = None
        self.btn_rt_start.setEnabled(True)
        if exit_code == 0:
            self.mi_status.setText("Status: Stopped")
            self.mi_status.setStyleSheet("background-color: #EBE9FF; color: #6C63FF; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
            self.rt_status.setText("Status: Stopped")
            self.rt_status.setStyleSheet("background-color: #EBE9FF; color: #6C63FF; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        else:
            self.mi_status.setText(f"Status: Error ({exit_code})")
            self.mi_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
            self.rt_status.setText(f"Status: Error ({exit_code})")
            self.rt_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        self.append_rt_log(f">>> [MI] Process exited with code {exit_code}")

    def start_blink(self) -> None:
        if self.blink_process and self.blink_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Already Running", "Blink detector is already running.")
            return

        self.blink_status.setText("Status: Starting...")
        self.blink_status.setStyleSheet("background-color: #FEF3C7; color: #92400E; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        self.btn_blink_start.setEnabled(False)
        self.append_blink_log("\n>>> [BLINK] STARTING BLINK DETECTOR")

        blink_args = [
            str(SCRIPTS / "blink_detector.py"),
            "--sfreq", self.blink_sfreq_input.text().strip() or "500",
            "--picks", self.blink_picks_input.text().strip() or "Fp1,Fp2",
            "--window", self.blink_window_input.text().strip() or "0.5",
            "--threshold-uv", self.blink_thr_input.text().strip() or "80",
            "--refractory", self.blink_refractory_input.text().strip() or "0.8",
        ]
        if self.blink_scale_uv_cb.isChecked():
            blink_args.append("--scale-to-uv")
        blink_key = self.blink_key_input.text().strip()
        if blink_key:
            blink_args.extend(["--key", blink_key])
        extra = self.blink_extra_input.text().strip()
        if extra:
            blink_args.extend(shlex.split(extra))

        self.blink_process = self._start_module_process("Blink", blink_args, self.append_blink_log, self._on_blink_finished)
        if self.blink_process:
            self.blink_status.setText("Status: Running")
            self.blink_status.setStyleSheet("background-color: #DCFCE7; color: #166534; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        else:
            self.blink_status.setText("Status: Error")
            self.blink_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
            self.btn_blink_start.setEnabled(True)

    def _on_blink_finished(self, exit_code: int) -> None:
        self.blink_process = None
        self.btn_blink_start.setEnabled(True)
        if exit_code == 0:
            self.blink_status.setText("Status: Stopped")
            self.blink_status.setStyleSheet("background-color: #EBE9FF; color: #6C63FF; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        else:
            self.blink_status.setText(f"Status: Error ({exit_code})")
            self.blink_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        self.append_blink_log(f">>> [Blink] Process exited with code {exit_code}")

    def start_gyro(self) -> None:
        if self.gyro_process and self.gyro_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "Already Running", "Gyro detector is already running.")
            return

        self.gyro_status.setText("Status: Starting...")
        self.gyro_status.setStyleSheet("background-color: #FEF3C7; color: #92400E; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        self.btn_gyro_start.setEnabled(False)
        self.append_gyro_log("\n>>> [GYRO] STARTING GYRO DETECTOR")

        gyro_args = [
            str(SCRIPTS / "gyro_detector.py"),
            "--sfreq", self.gyro_sfreq_input.text().strip() or "500",
            "--gyro-channels", self.gyro_channels_input.text().strip() or "5,6,7",
            "--stream-type", self.gyro_stream_type_input.text().strip() or "EEG",
            "--vel-forward", self.gyro_vel_f_input.text().strip() or "30",
            "--vel-backward", self.gyro_vel_b_input.text().strip() or "30",
            "--vel-left", self.gyro_vel_l_input.text().strip() or "100",
            "--vel-right", self.gyro_vel_r_input.text().strip() or "100",
            "--z-left-threshold", self.gyro_z_left_input.text().strip() or "20",
            "--z-right-threshold", self.gyro_z_right_input.text().strip() or "20",
            "--vel-return", self.gyro_vel_return_input.text().strip() or "120",
            "--deadzone-x", self.gyro_deadzone_x_input.text().strip() or "5",
            "--deadzone-y", self.gyro_deadzone_y_input.text().strip() or "20",
            "--deadzone-z", self.gyro_deadzone_z_input.text().strip() or "15",
            "--scale-factor", self.gyro_scale_input.text().strip() or "0.25",
            "--calibration-duration", self.gyro_calib_input.text().strip() or "2.0",
            "--smoothing-window", self.gyro_smoothing_input.text().strip() or "14",
            "--gamepad-repeat-interval", self.gyro_repeat_input.text().strip() or "0.40",
            "--key-mapping", self.gyro_keymap_input.text().strip() or "forward:w,backward:s,left:a,right:d",
        ]
        if self.gyro_use_z_lr_cb.isChecked():
            gyro_args.append("--use-z-for-lr")
        if self.gyro_gamepad_mode_cb.isChecked():
            gyro_args.append("--gamepad-mode")
        if self.gyro_output_keys_cb.isChecked():
            gyro_args.append("--output-keys")
        if self.gyro_verbose_cb.isChecked():
            gyro_args.append("--verbose")
        if self.gyro_drift_cb.isChecked():
            gyro_args.append("--enable-drift-correction")
        if self.gyro_invert_x_cb.isChecked():
            gyro_args.append("--invert-x")
        if self.gyro_invert_y_cb.isChecked():
            gyro_args.append("--invert-y")
        if self.gyro_invert_z_cb.isChecked():
            gyro_args.append("--invert-z")
        overlay_state_path = self.gyro_overlay_state_input.text().strip()
        if overlay_state_path:
            gyro_args.extend(["--overlay-state-file", overlay_state_path])
        extra = self.gyro_extra_input.text().strip()
        if extra:
            gyro_args.extend(shlex.split(extra))

        self.gyro_process = self._start_module_process("Gyro", gyro_args, self.append_gyro_log, self._on_gyro_finished)
        if self.gyro_process:
            self.gyro_status.setText("Status: Running")
            self.gyro_status.setStyleSheet("background-color: #DCFCE7; color: #166534; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        else:
            self.gyro_status.setText("Status: Error")
            self.gyro_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
            self.btn_gyro_start.setEnabled(True)

    def _on_gyro_finished(self, exit_code: int) -> None:
        self.gyro_process = None
        self.btn_gyro_start.setEnabled(True)
        if exit_code == 0:
            self.gyro_status.setText("Status: Stopped")
            self.gyro_status.setStyleSheet("background-color: #EBE9FF; color: #6C63FF; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        else:
            self.gyro_status.setText(f"Status: Error ({exit_code})")
            self.gyro_status.setStyleSheet("background-color: #FEE2E2; color: #991B1B; padding: 8px 18px; border-radius: 20px; font-weight: 700; font-size: 13px;")
        self.append_gyro_log(f">>> [Gyro] Process exited with code {exit_code}")

    def stop_realtime(self) -> None:
        if self.rt_process and self.rt_process.state() != QProcess.ProcessState.NotRunning:
            self.rt_status.setText("Status: Stopping...")
            self.mi_status.setText("Status: Stopping...")
            self.rt_process.terminate()
        else:
            self.rt_status.setText("Status: Idle")
            self.btn_rt_start.setEnabled(True)

    def stop_blink(self) -> None:
        if self.blink_process and self.blink_process.state() != QProcess.ProcessState.NotRunning:
            self.blink_status.setText("Status: Stopping...")
            self.blink_process.terminate()
        else:
            self.blink_status.setText("Status: Idle")
            self.btn_blink_start.setEnabled(True)

    def stop_gyro(self) -> None:
        if self.gyro_process and self.gyro_process.state() != QProcess.ProcessState.NotRunning:
            self.gyro_status.setText("Status: Stopping...")
            self.gyro_process.terminate()
        else:
            self.gyro_status.setText("Status: Idle")
            self.btn_gyro_start.setEnabled(True)

    def back_from_rt(self) -> None:
        self.stop_realtime()
        self.show_menu()

    def back_from_blink(self) -> None:
        self.stop_blink()
        self.show_menu()

    def back_from_gyro(self) -> None:
        self.stop_gyro()
        self.show_menu()

    def closeEvent(self, event) -> None:
        self.stop_realtime()
        self.stop_blink()
        self.stop_gyro()
        if self.master_status_timer.isActive():
            self.master_status_timer.stop()
        if self.master_process and self.master_process.state() != QProcess.ProcessState.NotRunning:
            self.master_process.terminate()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    win = EEGApp()
    win.show()
    sys.exit(app.exec())