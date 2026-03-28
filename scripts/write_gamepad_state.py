"""
Gamepad State Writer - Utility to write gamepad state for overlay display

This utility helps populate the JSON state file that gamepad_overlay.py reads.
You can use this to test the overlay, or integrate it into gyro_detector.py manually.

Usage:
  # Start overlay in one terminal
  python gamepad_overlay.py
  
  # Run this in another terminal to test
  python write_gamepad_state.py --command left --active-states left
  python write_gamepad_state.py --command forward --active-states forward
  python write_gamepad_state.py --command center
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional, List


def write_state(
    command: str,
    state_file: str = "gamepad_state.json",
    active_states: Optional[List[str]] = None,
    output_text: str = "idle",
):
    """Write gamepad state to JSON file."""
    
    if active_states is None:
        active_states = []
    
    state_data = {
        "command": command.lower(),
        "active_states": {
            "left": "left" in active_states,
            "right": "right" in active_states,
            "forward": "forward" in active_states,
            "backward": "backward" in active_states,
        },
        "output": output_text,
        "timestamp": __import__("time").time(),
    }
    
    state_path = Path(state_file)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Atomic write
    temp_file = state_path.with_suffix(".tmp")
    with open(temp_file, "w") as f:
        json.dump(state_data, f)
    
    temp_file.replace(state_path)
    
    print(f"✓ State written to {state_file}")
    print(f"  Command: {command}")
    print(f"  Active: {active_states}")
    print(f"  Output: {output_text}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Write gamepad state for overlay display",
        epilog="""
Examples:
  # Test left command
  python write_gamepad_state.py --command left
  
  # Test forward with active state
  python write_gamepad_state.py --command forward --active-states forward
  
  # Test multiple active states
  python write_gamepad_state.py --command center --active-states left right
  
  # Custom state file
  python write_gamepad_state.py --command left --state-file my_state.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--command",
        type=str,
        default="center",
        choices=["center", "left", "right", "forward", "backward", "calibrating"],
        help="Current gamepad command",
    )
    
    parser.add_argument(
        "--active-states",
        type=str,
        nargs="+",
        default=[],
        help="List of active direction states (left, right, forward, backward)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="idle",
        help="Output status text",
    )
    
    parser.add_argument(
        "--state-file",
        type=str,
        default="gamepad_state.json",
        help="Path to state JSON file",
    )
    
    args = parser.parse_args()
    
    # Validate active states
    valid_states = {"left", "right", "forward", "backward"}
    for state in args.active_states:
        if state not in valid_states:
            print(f"Error: Invalid state '{state}'. Must be one of: {valid_states}")
            sys.exit(1)
    
    write_state(
        command=args.command,
        state_file=args.state_file,
        active_states=args.active_states,
        output_text=args.output,
    )


if __name__ == "__main__":
    main()
