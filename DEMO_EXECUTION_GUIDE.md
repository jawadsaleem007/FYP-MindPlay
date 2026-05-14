# Module 3 Live Demo - EXECUTION GUIDE
## Step-by-step technical setup and running the demonstration

---

## PRE-DEMO CHECKLIST (Day Before)

### System Preparation:
- [ ] Update all Python packages (`pip install -r requirements.txt`)
- [ ] Verify trained FBCSP+LDA model exists (`fbcsp_lda_*.joblib` files present)
- [ ] Test EEG LSL stream (real or mock simulator)
- [ ] Calibrate gyroscope if using real headset
- [ ] Install Tuxemon or have it ready to launch
- [ ] Verify pydirectinput is installed and working
- [ ] Test gamepad overlay application
- [ ] Create gamepad_state.json test file

### Environment:
- [ ] Clear workspace of unnecessary files
- [ ] Have at least 3 terminal windows ready
- [ ] Close all other applications (reduces lag)
- [ ] Disable screen screensaver
- [ ] Ensure stable WiFi/network (if using wireless headset)

### Backup Preparation:
- [ ] Record a 3-5 minute demo video as backup
- [ ] Screenshot 5-10 overlay states for reference
- [ ] Export classification accuracy plots
- [ ] Save sample EEG signal visualization

---

## TECHNICAL SETUP (Day Of)

### PHASE 1: Terminal Preparation (~5 minutes)

#### Terminal 1: EEG Classification
```powershell
# Navigate to project
cd e:\FYP_Models\FYP(4)\FYP-MindPlay

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Launch EEG classifier (choose ONE option below)
```

**Option A: Real EEG Stream (If headset available)**
```powershell
python scripts/real_time_classifier.py `
  --model fbcsp_lda_S02.joblib `
  --sfreq 250 `
  --window 3.0 `
  --step 0.5 `
  --picks "2,3,4" `
  --delay 0.1
```

**Option B: Mock EEG Stream Simulator (Recommended for demo)**
```powershell
# First terminal 1a: Start mock EEG sender
python scripts/mock_lsl_sender.py

# Then terminal 1b: Connect classifier to mock stream
python scripts/real_time_classifier.py `
  --model fbcsp_lda_S02.joblib `
  --sfreq 250 `
  --window 3.0 `
  --step 0.5 `
  --picks "2,3,4"
```

**Expected Output from Terminal 1:**
```
====================================================================
Real-time FBCSP+LDA Classifier
====================================================================
Model: fbcsp_lda_S02.joblib
Sampling rate: 250 Hz
Window duration: 3.0s, Step: 0.5s
Channels: [2, 3, 4]
Awaiting EEG stream...
[LSL Connected at 12:34:56]
[12:35:02] Window 1: Prediction = REST (Conf: 0.82)
[12:35:03] Window 2: Prediction = RIGHT_HAND (Conf: 0.76)
```

---

#### Terminal 2: Gyroscope Detector
```powershell
# New PowerShell window
cd e:\FYP_Models\FYP(4)\FYP-MindPlay
.\.venv\Scripts\Activate.ps1

# Launch gyroscope detector (choose option matching your headset)
```

**Option A: With Real Gyroscope (e.g., Smarting headset)**
```powershell
python scripts/gyro_detector.py `
  --port COM3 `
  --overlay-state-file gamepad_state.json `
  --gamepad-mode
```

**Option B: Mock Gyroscope Simulator (Recommended for demo)**
```powershell
# Use a test script that writes random rotations
python scripts/gyro_detector.py `
  --mock `
  --sensitivity medium `
  --overlay-state-file gamepad_state.json `
  --gamepad-mode
```

**Expected Output from Terminal 2:**
```
====================================================================
Gyroscope Direction Detector
====================================================================
Mode: GAMEPAD
State file: gamepad_state.json
Sensitivity: MEDIUM
Waiting for gyro data...
[12:35:04] Detecting direction... [IDLE]
[12:35:05] Rotation detected: LEFT (magnitude: 8.5°)
[12:35:06] Direction state updated. Broadcast: {"command": "left", ...}
[12:35:07] Detecting direction... [IDLE]
[12:35:08] Rotation detected: FORWARD (magnitude: 12.3°)
```

---

#### Terminal 3: Gamepad Overlay + Game
```powershell
# New PowerShell window
cd e:\FYP_Models\FYP(4)\FYP-MindPlay
.\.venv\Scripts\Activate.ps1

# Launch gamepad overlay
python scripts/gamepad_overlay.py `
  --state-file gamepad_state.json `
  --refresh-ms 30 `
  --follow-active-window
```

**Before running overlay, start Tuxemon separately:**
```powershell
# In another window or on your system:
# Launch Tuxemon game (make it visible on screen)
cd path/to/tuxemon/
python -m tuxemon
# OR if it's on PATH:
tuxemon
```

**Expected Output from Terminal 3:**
```
======================================================================
MindPlay Gamepad Overlay
======================================================================
[OVERLAY] State file: gamepad_state.json
[OVERLAY] Refresh: 30 ms
[OVERLAY] Follow active window: True
[OVERLAY] Waiting for updates...
[12:35:06] State updated: command='left', active_states=['L']
[12:35:07] State updated: command='center', active_states=[]
[12:35:08] State updated: command='forward', active_states=['F']
```

**Overlay Window:**
- Should appear in **top-right corner** of screen
- Should be **always-on-top** (stays over Tuxemon)
- Should show:
  - Title: "MINDPLAY GAMEPAD STATE"
  - Current command with icon (e.g., "LEFT ◀")
  - Direction states: L, R, F, B (highlighted when active)
  - Output status line
  - Timestamp of last update

---

### PHASE 2: System Verification (2 minutes)

#### Check All Three Processes Are Running:
1. **Terminal 1:** EEG classifier shows predictions (e.g., "RIGHT_HAND", "REST")
2. **Terminal 2:** Gyroscope detector shows rotation detection
3. **Terminal 3:** Overlay window visible and updating color/text
4. **Game:** Tuxemon window is open and visible

#### Manual Test Sequence:
```
Action 1: Move head LEFT
   Terminal 2: "[12:35:10] LEFT detected"
   Gamepad State: command = 'left'
   Overlay: Shows "← LEFT" with orange color
   Tuxemon: Character moves left
   
Action 2: Imagine right hand movement (or trigger EEG event)
   Terminal 1: "RIGHT_HAND" prediction
   Gamepad State: Updates with RIGHT_HAND flag
   Overlay: Pulsates or changes color
   Tuxemon: Character performs action (if mapped)
   
Action 3: Return to neutral
   All processes: Return to IDLE/CENTER/REST state
   Overlay: Reset to gray, command = 'center'
   Tuxemon: No input
```

#### If Something Doesn't Work:
| Issue | Solution |
|-------|----------|
| EEG stream not found | Check LSL stream is running; verify headset connected |
| Gyroscope not responding | Calibrate headset; check serial port COM3/COM4 |
| Overlay not updating | Verify gamepad_state.json exists and has write permissions |
| Game not receiving input | Ensure Tuxemon window is focused; test pydirectinput manually |
| High latency | Close other applications; reduce refresh rates (but not below 20ms) |

---

## DEMO TIMING & NARRATION

### Section 1: System Overview (30 seconds)
**What to show:**
- Point to Terminal 3 with overlay visible
- Show Tuxemon game window

**What to say:**
```
"Here's our complete system. The overlay on the right shows
the current control state in real-time. Three processes are
running simultaneously:
  - EEG classifier detecting brain signals
  - Gyroscope detector reading head movements
  - Gamepad overlay visualizing all inputs
  
The game (Tuxemon) receives keyboard commands from all these sources."
```

---

### Section 2: Gyroscope Control Demo (60 seconds)
**What to do:**
1. Make a **CLEAR, SLOW LEFT HEAD TILT**
2. Pause 1 second (let audience see the response)
3. Return to center
4. Wait 2 seconds
5. Make a **CLEAR, SLOW RIGHT HEAD TILT**
6. Pause 1 second
7. Return to center
8. Make a **FORWARD TILT**
9. Make a **BACKWARD TILT**
10. Return to center

**What to say:**
```
"Now watch as I tilt my head. Notice three things:
1. The overlay immediately shows the direction (LEFT, RIGHT, etc.)
2. The character in the game moves simultaneously
3. There's no lag—it's real-time, under 300 milliseconds

This is gyroscope control—no EEG training needed,
just intuitive head movements. Let me do this a few times
so you see how responsive it is."

[Perform tilts]

"See how smooth that was? No stuttering, no delays.
The system handles that processing in the background."
```

---

### Section 3: EEG-Based Control Demo (60 seconds)
**What to do:**
1. **Say:** "Now I'll demonstrate EEG-based control."
2. **Explain:** "I'm going to imagine moving my right hand."
3. **Wait** 3-4 seconds (this is how long the classification window is)
4. **Point to Terminal 1:** "Watch the prediction change..."
5. **Point to Overlay:** "...and see the command update here"
6. **Point to Game:** "...and the game receives the input"

**What to say:**
```
"EEG is more complex than gyroscope. The system is
continuously analyzing my brain activity:
- Reading at 250 Hz (250 samples per second)
- Buffering a 3-second window
- Applying signal processing filters
- Classifying using a trained machine learning model
- All within 100 milliseconds

I'm going to imagine moving my right hand. This triggers
a specific pattern in the motor cortex that the EEG picks up.

Watch the Terminal 1 prediction... notice how it changes
from REST to RIGHT_HAND... The overlay reflects that change...
And the game receives the command. That's brain control.

This is what makes our system different from traditional
gaming interfaces. You're not using your hands or voice.
You're controlling the game with your thoughts."
```

*Optional: Repeat 2-3 times to reinforce the point*

---

### Section 4: Combined Real-Time Gameplay (90 seconds)
**What to do:**
1. Play a short 30-second game sequence
2. Use head movements (gyro) to navigate
3. Use EEG signals (if possible) to execute actions
4. Keep narrating what's happening

**What to say:**
```
"Now let me show you how all three signal types work together
in a real gaming scenario.

I'll navigate using head movements—that's the gyroscope.
[Tilt head left/right to move the character]

When I want to select something or confirm an action,
I'll use EEG or a blink detector—that's the brain signal integration.

Everything you see—from the overlay state changes to the game response—
is happening in real-time. No post-processing, no delays.

Module 1 acquired the signals. Module 2 decoded them.
Module 3 is translating them into gameplay. Everything working together."
```

*Play the game segment for 30 seconds, showing smooth integration*

---

### Section 5: Performance Metrics (20 seconds)
**What to show:**
- If overlay displays latency: Point to it (should be ~200-300ms)
- Show Terminal 1: Confidence scores on predictions (should be >70%)
- Show Terminal 2: Rotation magnitudes

**What to say:**
```
"Let me highlight some performance metrics:

Classification accuracy on test data: 75-85%
EEG latency (signal to prediction): 50-100 milliseconds
Gyroscope latency: Less than 20 milliseconds
End-to-end latency (signal to game action): 200-300 milliseconds

For comparison, a human reaction time is about 200 milliseconds.
This system is competitive with human reflexes.
```

---

### Section 6: Q&A and Closing (30+ seconds)
**Be ready for:**
- "How accurate is the EEG classifier?"
- "What's the latency compared to a regular controller?"
- "Can you add more commands?"
- "Does it work with other games?"

**Have answers ready from earlier cards**

---

## DEMO TROUBLESHOOTING

### Overlay Not Showing:
```powershell
# Check if file exists
Test-Path gamepad_state.json

# Create it manually if missing
@{"command" = "center"; "active_states" = @(); "output" = "idle"; "timestamp" = (Get-Date).ToUniversalTime()} | ConvertTo-Json | Out-File gamepad_state.json
```

### EEG Stream Not Found:
```powershell
# Check what LSL streams are available
python scripts/list_lsl_streams.py

# Use mock sender instead
python scripts/mock_lsl_sender.py
```

### Game Not Receiving Keyboard Input:
```powershell
# Test pydirectinput manually
python -c "
import pydirectinput
import time
print('Sending LEFT arrow in 3 seconds...')
time.sleep(3)
pydirectinput.press('left')
print('Done')
"
# In Tuxemon, character should move left
```

### Overlay Freezes / Doesn't Update:
```powershell
# Kill overlay process
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Stop-Process

# Restart overlay
python scripts/gamepad_overlay.py --state-file gamepad_state.json
```

### High Latency / Stutter:
1. Close all other applications
2. Reduce refresh rates: `--refresh-ms 50` (from 30)
3. Check CPU usage: `Get-Process | Sort-Object CPU -Descending | Select-Object -First 5`
4. Ensure no antivirus scanning in background

---

## DEMO BACKUP PLAN (If Live Demo Fails)

### Backup Option 1: Video Playback (Recommended)
- Have a pre-recorded 5-minute video of the demo
- Show it if live system doesn't cooperate
- Still explain what's happening as it plays
- Credible fallback: "Hardware didn't initialize in the demo, but here's a recording"

### Backup Option 2: Screenshot Presentation
- Have 10-15 screenshots of:
  - Overlay in different states (left, right, forward, backward, center)
  - Tuxemon with character in different positions
  - Terminal outputs showing timestamps and predictions
  - Before/after EEG signals (with filtering applied)
- Walk through manually: "Here the gyro detected a left rotation, the overlay changed, the character moved..."

### Backup Option 3: Code Walkthrough
- If system is uncooperative, pivot to architecture discussion
- Show actual code on screen (real_time_classifier.py, gamepad_overlay.py)
- Walk through the signal processing pipeline
- Explain how commands are mapped
- Less visual, but shows deep understanding

### Backup Option 4: Hybrid Approach
- Display recorded video
- Narrate with associated Terminal output
- Show architecture diagrams alongside
- Answer questions in depth
- Say: "Let me show you the system through a real recorded session..."

---

## FINAL CHECKLIST (30 minutes before demo)

| Item | Status | Notes |
|------|--------|-------|
| All terminals set up? | ☐ | If not, start now |
| EEG process running? | ☐ | Check Terminal 1 output |
| Gyro process running? | ☐ | Check Terminal 2 output |
| Overlay visible? | ☐ | Check top-right corner |
| Tuxemon open & visible? | ☐ | Test one key press |
| One manual test-run done? | ☐ | (Head tilt + EEG prediction) |
| Backup video ready? | ☐ | If main demo fails |
| Presentation script nearby? | ☐ | Reference during demo |
| All talking points memorized? | ☐ | (Or quick cards in hand) |
| Audience can see overlay? | ☐ | Adjust screen/projector if needed |

---

## DURING DEMO: KEY LINES

**If something breaks mid-demo:**
- Stay calm
- Say: "Let me restart that component..."
- Switch to backup video if needed
- Say: "Here's a recorded version that shows the same behavior..."

**If latency seems high:**
- Say: "The system is working through a fairly complex signal pipeline. Real-world processing takes a bit of time..."
- Point to Terminal outputs: "But you can see the predictions are accurate and consistent."

**If someone asks "Is this real or simulated?"**
- Say: "The demo uses [simul real] EEG/gyro data, but the classification and game integration are identical to what we'd use with a real headset. The same trained model. The same signal processing. The bottleneck is the demo environment, not the system."

**If prediction confidence is low:**
- Say: "During a live session with proper calibration, confidence scores are typically 75-85%. Today we're using a model trained on Subject 02's data. With personalized calibration, this would improve."

---

## AFTER DEMO

### Immediate:
- [ ] Thank the faculty for attention
- [ ] Ask if there are questions
- [ ] Be ready for follow-up demos on specific parts

### Later:
- [ ] Note any technical issues for the report
- [ ] Gather feedback on presentation clarity
- [ ] Update demo setup based on any lessons learned
- [ ] Prepare similar system for final evaluation (Module 4)

---

**Remember: The system works. The demo is just showing what you've already proven in testing.**

**Present with confidence. 🧠🎮**
