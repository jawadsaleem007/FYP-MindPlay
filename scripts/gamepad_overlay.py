"""Native Windows gamepad overlay for gyro_detector state output."""

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
from typing import Any, Dict

import wx

try:
    from scripts.command_cooldown import cooldown_remaining_from_state
except ImportError:
    from command_cooldown import cooldown_remaining_from_state


HWND_TOPMOST = -1
SWP_NOSIZE = 0x0001
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040


class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class GamepadOverlay(wx.Frame):
    """Always-on-top overlay showing current gamepad direction/state."""

    def __init__(self, state_file: Path, refresh_ms: int, follow_active_window: bool):
        wx.Frame.__init__(
            self,
            None,
            title="MindPlay Gamepad Overlay",
            style=wx.CAPTION | wx.STAY_ON_TOP | wx.FRAME_TOOL_WINDOW,
        )

        self.state_file = state_file.resolve()
        self.refresh_ms = max(20, refresh_ms)
        self.follow_active_window = follow_active_window
        self.last_state: Dict[str, Any] = {}
        self.cooldown_was_active = False
        self.update_count = 0
        self.user32 = ctypes.windll.user32
        self.overlay_hwnd = 0

        self.overlay_width = 420
        self.overlay_height = 282
        self.margin = 12

        self.SetSize((self.overlay_width, self.overlay_height))
        self.SetBackgroundColour(wx.Colour(255, 255, 255))

        self._create_ui()
        self._position_top_right_screen()

        self.Bind(wx.EVT_CLOSE, self._on_close)

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer)
        self.timer.Start(self.refresh_ms)

        print("=" * 70)
        print("MindPlay Gamepad Overlay")
        print("=" * 70)
        print(f"[OVERLAY] State file: {self.state_file}")
        print(f"[OVERLAY] Refresh: {self.refresh_ms} ms")
        print(f"[OVERLAY] Follow active window: {self.follow_active_window}")
        print("[OVERLAY] Waiting for updates...")

    def _create_ui(self) -> None:
        panel = wx.Panel(self)
        panel.SetBackgroundColour(wx.Colour(255, 255, 255))

        root = wx.BoxSizer(wx.VERTICAL)

        title = wx.StaticText(panel, label="MINDPLAY GAMEPAD STATE")
        title.SetFont(wx.Font(13, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        title.SetForegroundColour(wx.Colour(0, 0, 0))
        root.Add(title, 0, wx.ALL, 10)

        divider = wx.StaticLine(panel)
        root.Add(divider, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        cmd_label = wx.StaticText(panel, label="Current Command")
        cmd_label.SetForegroundColour(wx.Colour(70, 70, 70))
        root.Add(cmd_label, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        self.command_text = wx.StaticText(panel, label="● WAITING")
        self.command_text.SetFont(wx.Font(26, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.command_text.SetForegroundColour(wx.Colour(120, 120, 120))
        root.Add(self.command_text, 0, wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        dir_label = wx.StaticText(panel, label="Direction Latches")
        dir_label.SetForegroundColour(wx.Colour(70, 70, 70))
        root.Add(dir_label, 0, wx.LEFT | wx.RIGHT, 10)

        dir_row = wx.BoxSizer(wx.HORIZONTAL)
        self.dir_widgets: Dict[str, wx.StaticText] = {}
        for key, short in (("left", "L"), ("right", "R"), ("forward", "F"), ("backward", "B")):
            pill = wx.StaticText(panel, label=f" {short} ", style=wx.ALIGN_CENTER_HORIZONTAL | wx.BORDER_SIMPLE)
            pill.SetFont(wx.Font(12, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
            pill.SetForegroundColour(wx.Colour(120, 120, 120))
            pill.SetBackgroundColour(wx.Colour(230, 230, 230))
            pill.SetMinSize((58, 30))
            self.dir_widgets[key] = pill
            dir_row.Add(pill, 0, wx.RIGHT, 8)
        root.Add(dir_row, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        self.output_text = wx.StaticText(panel, label="Output: idle")
        self.output_text.SetForegroundColour(wx.Colour(80, 80, 80))
        root.Add(self.output_text, 0, wx.LEFT | wx.RIGHT | wx.TOP, 10)

        self.cooldown_text = wx.StaticText(panel, label="Cooldown: ready")
        self.cooldown_text.SetForegroundColour(wx.Colour(80, 140, 80))
        root.Add(self.cooldown_text, 0, wx.LEFT | wx.RIGHT | wx.TOP, 8)

        self.status_text = wx.StaticText(panel, label="Status: waiting for state file updates")
        self.status_text.SetForegroundColour(wx.Colour(120, 120, 120))
        root.Add(self.status_text, 0, wx.LEFT | wx.RIGHT | wx.TOP | wx.BOTTOM, 10)

        panel.SetSizer(root)
        panel.Layout()

    def _position_top_right_screen(self) -> None:
        display_w, _display_h = wx.GetDisplaySize()
        pos_x = max(0, display_w - self.overlay_width - self.margin)
        pos_y = self.margin
        self.SetPosition((pos_x, pos_y))

    def _pin_to_top_right_of_active_window(self) -> None:
        if not self.follow_active_window:
            self._position_top_right_screen()
            return

        try:
            if not self.overlay_hwnd:
                self.overlay_hwnd = int(self.GetHandle())

            fg_hwnd = self.user32.GetForegroundWindow()
            if fg_hwnd and fg_hwnd != self.overlay_hwnd:
                rect = RECT()
                if self.user32.GetWindowRect(fg_hwnd, ctypes.byref(rect)):
                    pos_x = rect.right - self.overlay_width - self.margin
                    pos_y = rect.top + self.margin
                    pos_x = max(0, pos_x)
                    pos_y = max(0, pos_y)
                    self.SetPosition((pos_x, pos_y))
        except Exception:
            self._position_top_right_screen()

        try:
            self.user32.SetWindowPos(
                int(self.GetHandle()),
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            )
        except Exception:
            pass

    def _load_state(self) -> Dict[str, Any]:
        default_state = {
            "command": "center",
            "active_states": {
                "left": False,
                "right": False,
                "forward": False,
                "backward": False,
            },
            "output": "idle",
            "cooldown_until": 0.0,
            "cooldown_seconds": 0.0,
            "cooldown_source": "",
        }

        try:
            if self.state_file.exists():
                with open(self.state_file, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return data
        except Exception as exc:
            self.status_text.SetLabel(f"Status: read error ({exc})")
        return default_state

    def _render_state(self, state: Dict[str, Any]) -> None:
        command = str(state.get("command") or "center").lower()

        color_map = {
            "center": wx.Colour(120, 120, 120),
            "left": wx.Colour(220, 120, 50),
            "right": wx.Colour(220, 120, 50),
            "forward": wx.Colour(40, 160, 85),
            "backward": wx.Colour(220, 60, 60),
            "calibrating": wx.Colour(160, 90, 230),
        }
        icon_map = {
            "center": "●",
            "left": "◀",
            "right": "▶",
            "forward": "▲",
            "backward": "▼",
            "calibrating": "⚙",
        }

        self.command_text.SetLabel(f"{icon_map.get(command, '●')} {command.upper()}")
        self.command_text.SetForegroundColour(color_map.get(command, wx.Colour(120, 120, 120)))

        active = state.get("active_states", {}) or {}
        for key, widget in self.dir_widgets.items():
            if bool(active.get(key, False)):
                widget.SetBackgroundColour(wx.Colour(114, 214, 255))
                widget.SetForegroundColour(wx.Colour(0, 0, 0))
            else:
                widget.SetBackgroundColour(wx.Colour(230, 230, 230))
                widget.SetForegroundColour(wx.Colour(120, 120, 120))
            widget.Refresh()

        output = str(state.get("output", "idle"))
        self.output_text.SetLabel(f"Output: {output}")

        cooldown_remaining = cooldown_remaining_from_state(state)
        if cooldown_remaining > 0.0:
            source = str(state.get("cooldown_source") or "gyro")
            self.cooldown_text.SetLabel(f"Cooldown: {cooldown_remaining:.1f}s blocking blink/MI ({source})")
            self.cooldown_text.SetForegroundColour(wx.Colour(190, 85, 45))
            self.cooldown_was_active = True
        else:
            self.cooldown_text.SetLabel("Cooldown: ready")
            self.cooldown_text.SetForegroundColour(wx.Colour(80, 140, 80))
            self.cooldown_was_active = False

        active_dirs = [k for k, v in active.items() if v]
        self.status_text.SetLabel(
            f"Status: updates={self.update_count} active={active_dirs if active_dirs else ['none']}"
        )
        self.Layout()

    def _on_timer(self, _event: wx.Event) -> None:
        self._pin_to_top_right_of_active_window()

        state = self._load_state()
        cmp_state = dict(state)
        cmp_state.pop("timestamp", None)
        last_cmp = dict(self.last_state)
        last_cmp.pop("timestamp", None)
        cooldown_active = cooldown_remaining_from_state(state) > 0.0
        if cmp_state == last_cmp and not cooldown_active and not self.cooldown_was_active:
            return

        state_changed = cmp_state != last_cmp
        if state_changed:
            self.update_count += 1
        self._render_state(state)
        if state_changed:
            self.last_state = state
            print(f"[OVERLAY] Update {self.update_count}: command={state.get('command', 'center')}")

    def _on_close(self, _event: wx.Event) -> None:
        if self.timer.IsRunning():
            self.timer.Stop()
        self.Destroy()


class GamepadApp(wx.App):
    def __init__(self, state_file: Path, refresh_ms: int, follow_active_window: bool):
        self._state_file = state_file
        self._refresh_ms = refresh_ms
        self._follow_active_window = follow_active_window
        self.frame = None
        super().__init__()

    def OnInit(self) -> bool:
        self.frame = GamepadOverlay(
            state_file=self._state_file,
            refresh_ms=self._refresh_ms,
            follow_active_window=self._follow_active_window,
        )
        self.frame.Show(True)
        self.SetTopWindow(self.frame)
        return True


def _resolve_default_state_file() -> Path:
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "gamepad_state.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="MindPlay native gamepad state overlay")
    parser.add_argument(
        "--state-file",
        type=str,
        default="",
        help="Path to the JSON state file written by gyro_detector",
    )
    parser.add_argument(
        "--refresh-ms",
        type=int,
        default=50,
        help="Overlay refresh interval in milliseconds",
    )
    parser.add_argument(
        "--follow-active-window",
        action="store_true",
        help="Pin overlay to the top-right corner of the active window",
    )
    parser.add_argument(
        "--no-follow-active-window",
        action="store_true",
        help="Keep overlay pinned to screen top-right instead of active-window top-right",
    )

    args = parser.parse_args()
    state_file = Path(args.state_file).resolve() if args.state_file else _resolve_default_state_file()
    follow_active_window = True
    if args.no_follow_active_window:
        follow_active_window = False
    if args.follow_active_window:
        follow_active_window = True

    app = GamepadApp(
        state_file=state_file,
        refresh_ms=args.refresh_ms,
        follow_active_window=follow_active_window,
    )
    app.MainLoop()


if __name__ == "__main__":
    main()
