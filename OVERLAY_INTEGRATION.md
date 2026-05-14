"""
GAMEPAD OVERLAY INTEGRATION GUIDE

The overlay system consists of two parts:
1. gamepad_overlay.py - Displays the overlay window (top-right corner, always on top)
2. State file - JSON file that contains current gamepad state

QUICK START - Test Mode
========================

Terminal 1 - Start the overlay:
  cd e:\FYP_Models\FYP(4)\FYP-MindPlay
  python scripts/gamepad_overlay.py

Terminal 2 - Update state (simulate commands):
  cd e:\FYP_Models\FYP(4)\FYP-MindPlay
  python scripts/write_gamepad_state.py --command left
  python scripts/write_gamepad_state.py --command forward --active-states forward
  python scripts/write_gamepad_state.py --command center


INTEGRATION WITH GYRO DETECTOR
================================

To integrate with gyro_detector.py, add this code to the main loop:

1. Add import at top of gyro_detector.py:
   import json
   from pathlib import Path

2. Create an OverlayWriter near the start of main():
   class OverlayStateWriter:
       def __init__(self, state_file):
           self.state_file = Path(state_file) if state_file else None
       
       def update(self, command, active_states, output_text="idle"):
           if not self.state_file:
               return
           try:
               state = {
                   "command": command or "center",
                   "active_states": active_states,
                   "output": output_text,
                   "timestamp": time.time(),
               }
               temp_file = self.state_file.with_suffix('.tmp')
               with open(temp_file, 'w') as f:
                   json.dump(state, f)
               temp_file.replace(self.state_file)
           except:
               pass

3. Add argument to argparse:
   ap.add_argument('--overlay-state-file', type=str, default='',
                   help='JSON file for overlay state updates')

4. Create writer in main():
   overlay_writer = OverlayStateWriter(args.overlay_state_file) if args.overlay_state_file else None

5. Update overlay in the detection loop:
   if overlay_writer:
       overlay_writer.update(
           command=detector.current_direction if args.gamepad_mode else direction,
           active_states=detector.direction_active,
           output_text="idle"
       )

6. Run with both:
   Terminal 1:
     python scripts/gamepad_overlay.py --state-file gamepad_state.json
   
   Terminal 2:
     python scripts/gyro_detector.py --overlay-state-file gamepad_state.json [other args...]


OVERLAY FEATURES
=================

Display Elements:
  • Command indicator: Shows current command with icon (●, ◀, ▶, ▲, ▼, ⚙)
  • Direction states: L, R, F, B indicators (bright when active)
  • Output status: Shows what action is happening
  • Colors: Orange (left/right), Green (forward), Red (backward), Gray (center)

Window:
  • Always on top (stays visible over games and other apps)
  • 95% opacity (slightly transparent)
  • Top-right corner positioning
  • No window decorations (clean look like Discord overlay)
  • Frameless with minimize/close

Customization:
  python gamepad_overlay.py --refresh-ms 30    # Faster updates
  python gamepad_overlay.py --state-file my_state.json


STATE FILE FORMAT
==================

JSON structure written by gyro_detector.py:

{
  "command": "left",
  "active_states": {
    "left": true,
    "right": false,
    "forward": false,
    "backward": false
  },
  "output": "holding left",
  "cooldown_until": 0.0,
  "cooldown_seconds": 0.0,
  "cooldown_source": "",
  "timestamp": 1711612000.123
}

Fields:
  command (str): Current active command - "center", "left", "right", "forward", "backward", "calibrating"
  active_states (dict): Boolean flags for each direction
  output (str): Status text - "idle", "holding left", "pressing up", etc.
  cooldown_until (float): Unix timestamp until blink/MI commands are ignored
  cooldown_seconds (float): Configured cooldown duration in seconds
  cooldown_source (str): Gyro direction that started the cooldown
  timestamp (float): Time when state was written


EXAMPLE INTEGRATION (Minimal)
==============================

Add to gyro_detector.py:

import json
from pathlib import Path

class OverlayWriter:
    def __init__(self, state_file):
        self.state_file = Path(state_file) if state_file else None
    
    def update(self, command, active_states):
        if not self.state_file:
            return
        try:
            with open(self.state_file.with_suffix('.tmp'), 'w') as f:
                json.dump({
                    "command": command or "center",
                    "active_states": active_states,
                }, f)
            self.state_file.with_suffix('.tmp').replace(self.state_file)
        except:
            pass

# In main():
overlay = OverlayWriter(args.overlay_state_file)

# In detection loop:
overlay.update(detector.current_direction, detector.direction_active)

# Run:
python gyro_detector.py ... --overlay-state-file gamepad_state.json

# In separate terminal:
python gamepad_overlay.py
"""
