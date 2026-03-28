"""
Web-based Gamepad Overlay - Flask app
Displays gamepad state in browser window (top-right corner)

No tkinter/tcl required!

Run:
  python gamepad_overlay_web.py
  
Then open browser to:
  http://localhost:5000
"""

from flask import Flask, render_template_string, jsonify
import json
from pathlib import Path
import threading
import webbrowser
import time

app = Flask(__name__)

STATE_FILE = "gamepad_state.json"


def load_state():
    """Load gamepad state from JSON file."""
    default_state = {
        "command": "center",
        "active_states": {
            "left": False,
            "right": False,
            "forward": False,
            "backward": False,
        },
        "output": "idle",
    }
    
    try:
        state_path = Path(STATE_FILE)
        if state_path.exists():
            with open(state_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    
    return default_state


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MindPlay Gamepad Overlay</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0F172A;
            display: flex;
            justify-content: flex-end;
            align-items: flex-start;
            min-height: 100vh;
            padding: 20px;
        }
        
        .overlay {
            width: 340px;
            background: linear-gradient(135deg, #0F172A 0%, #1A1F3A 100%);
            border: 1px solid #1E293B;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
        }
        
        .header {
            font-size: 11px;
            font-weight: 600;
            color: #E0E7FF;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            padding-bottom: 10px;
            border-bottom: 1px solid #1E293B;
        }
        
        .command-section {
            margin-bottom: 16px;
        }
        
        .command-label {
            font-size: 10px;
            color: #94A3B8;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        
        .command-display {
            font-size: 24px;
            font-weight: bold;
            color: #64748B;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .command-display.left {
            color: #F97316;
        }
        
        .command-display.right {
            color: #F97316;
        }
        
        .command-display.forward {
            color: #10B981;
        }
        
        .command-display.backward {
            color: #EF4444;
        }
        
        .command-display.calibrating {
            color: #A78BFA;
        }
        
        .direction-label {
            font-size: 10px;
            color: #94A3B8;
            text-transform: uppercase;
            margin: 12px 0 8px 0;
        }
        
        .direction-buttons {
            display: flex;
            gap: 8px;
        }
        
        .dir-btn {
            flex: 1;
            padding: 8px;
            background: #1E293B;
            color: #64748B;
            border: 1px solid #334155;
            border-radius: 4px;
            font-weight: bold;
            font-size: 11px;
            text-align: center;
            transition: all 0.2s ease;
            cursor: default;
        }
        
        .dir-btn.active {
            background: #0891B2;
            color: #ECFEFF;
            border-color: #06B6D4;
        }
        
        .output-section {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #1E293B;
            font-size: 10px;
            color: #9CA3AF;
        }
        
        .output-label {
            display: inline;
            color: #94A3B8;
        }
    </style>
</head>
<body>
    <div class="overlay">
        <div class="header">MINDPLAY GAMEPAD</div>
        
        <div class="command-section">
            <div class="command-label">Command</div>
            <div class="command-display" id="commandDisplay">
                <span id="commandIcon">●</span>
                <span id="commandText">CENTER</span>
            </div>
        </div>
        
        <div class="direction-label">Direction State</div>
        <div class="direction-buttons">
            <div class="dir-btn" id="btnL">L</div>
            <div class="dir-btn" id="btnR">R</div>
            <div class="dir-btn" id="btnF">F</div>
            <div class="dir-btn" id="btnB">B</div>
        </div>
        
        <div class="output-section">
            <span class="output-label">Output:</span>
            <span id="outputText"> idle</span>
        </div>
    </div>
    
    <script>
        const commandDisplay = document.getElementById('commandDisplay');
        const commandIcon = document.getElementById('commandIcon');
        const commandText = document.getElementById('commandText');
        const outputText = document.getElementById('outputText');
        const dirBtns = {
            left: document.getElementById('btnL'),
            right: document.getElementById('btnR'),
            forward: document.getElementById('btnF'),
            backward: document.getElementById('btnB'),
        };
        
        const iconMap = {
            center: '●',
            left: '◀',
            right: '▶',
            forward: '▲',
            backward: '▼',
            calibrating: '⚙',
        };
        
        async function updateState() {
            try {
                const response = await fetch('/api/state');
                const state = await response.json();
                
                const command = (state.command || 'center').toLowerCase();
                commandIcon.textContent = iconMap[command] || '●';
                commandText.textContent = command.toUpperCase();
                commandDisplay.className = `command-display ${command}`;
                
                outputText.textContent = ` ${state.output || 'idle'}`;
                
                // Update direction buttons
                for (const [dir, btn] of Object.entries(dirBtns)) {
                    if (state.active_states[dir]) {
                        btn.classList.add('active');
                    } else {
                        btn.classList.remove('active');
                    }
                }
            } catch (error) {
                console.error('Error fetching state:', error);
            }
        }
        
        // Update every 50ms
        setInterval(updateState, 50);
        updateState();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/state')
def get_state():
    return jsonify(load_state())


def open_browser():
    """Open browser after a short delay."""
    time.sleep(1)
    webbrowser.open('http://localhost:5000')


if __name__ == '__main__':
    print("=" * 60)
    print("MindPlay Gamepad Overlay (Web)")
    print("=" * 60)
    print("Starting server...")
    print("Opening browser in 1 second...")
    print("\nBrowser URL: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    # Open browser in background thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run Flask app
    app.run(host='127.0.0.1', port=5000, debug=False)
