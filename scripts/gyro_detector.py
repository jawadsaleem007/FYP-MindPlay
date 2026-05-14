"""Real-time gyroscope velocity-based detector for Smarting24 LSL stream.

This script connects to an LSL stream containing gyroscope data (3-axis angular velocity),
detects head movements (left/right/forward/backward), and sends WASD key presses.

SMARTING24 DEVICE ORIENTATION (worn on back of head):
- Dimensions: ~50mm x 40mm x 15mm 
- When worn normally with USB port facing down:
  - X-axis: Roll (head tilt left/right - ear to shoulder)
  - Y-axis: Pitch (head tilt forward/backward - chin to chest)  
  - Z-axis: Yaw (head rotation left/right - looking left/right)

VELOCITY-BASED DETECTION:
- Uses angular velocity directly (deg/s) instead of integrated angles
- Prevents re-triggering: Once a direction is detected, it won't trigger again
  until velocity drops below return threshold (prevents spam)
- Baseline calibration removes drift/bias

WASD MAPPING (customizable):
  W = Forward (head tilt forward, Y-axis positive velocity)
  S = Backward (head tilt backward, Y-axis negative velocity)
  A = Left (head tilt left, X-axis negative velocity)
  D = Right (head tilt right, X-axis positive velocity)

Example usage:
  python scripts\\gyro_detector.py --vel-forward 80 --vel-back 80 --vel-left 100 --vel-right 100 --vel-return 20 --scale-factor 1.0

All parameters are customizable via command line arguments.
--vel-forward 80      # Forward threshold (Y+ velocity, deg/s)
--vel-backward 80     # Backward threshold (Y- velocity, deg/s)
--vel-left 100        # Left threshold (X- velocity, deg/s)
--vel-right 100       # Right threshold (X+ velocity, deg/s)
--vel-return 20       # Must drop below this to unlock
--scale-factor 1.0    # Convert raw to deg/s
--calibration-duration 2.0
--smoothing-window 5  # Samples to average
--key-mapping "forward:w,backward:s,left:a,right:d"
--output-keys         # Actually send keypresses
--verbose             # Show velocity details
"""
import time
import argparse
import ctypes
import atexit
import json
from pathlib import Path
import subprocess
import sys
import os
import importlib
from typing import List, Tuple, Optional, Union
from collections import deque
from datetime import datetime

import numpy as np
from pylsl import StreamInlet, resolve_byprop

try:
    from scripts.command_cooldown import cooldown_payload, resolve_state_file
except ImportError:
    from command_cooldown import cooldown_payload, resolve_state_file

try:
    pydirectinput = importlib.import_module('pydirectinput')
    _key_output_available = True
except Exception:
    pydirectinput = None
    _key_output_available = False
    print("Warning: pydirectinput not available. Install with: pip install pydirectinput")

try:
    from pynput.keyboard import Listener
    _listener_available = True
except Exception:
    Listener = None
    _listener_available = False
    print("Warning: pynput not available. Recalibration hotkeys (R/Q) disabled.")


def _is_windows_admin() -> bool:
    """Return True when running elevated on Windows."""
    if os.name != 'nt':
        return True
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def ensure_admin_privileges() -> None:
    """On Windows, re-launch script with UAC prompt unless already elevated."""
    if os.name != 'nt' or _is_windows_admin():
        return

    print("Requesting administrator privileges (UAC)...")
    params = subprocess.list2cmdline(sys.argv)
    result = ctypes.windll.shell32.ShellExecuteW(None, 'runas', sys.executable, params, None, 1)
    if result <= 32:
        raise RuntimeError("Failed to relaunch with administrator privileges.")
    sys.exit(0)


class VelocityDetector:
    """Velocity-based gyroscope detector with hysteresis to prevent re-triggering."""
    
    def __init__(self, 
                 # Velocity thresholds for each direction (deg/s)
                 vel_forward: float = 120.0,
                 vel_backward: float = 120.0,
                 vel_left: float = 80.0,
                 vel_right: float = 80.0,
                 z_left_threshold: float = 80.0,
                 z_right_threshold: float = 80.0,
                 use_z_for_lr: bool = False,
                 y_like_lr: bool = False,
                 gamepad_mode: bool = False,
                 gamepad_lr_only: bool = False,
                 gamepad_ud_only: bool = False,
                 gamepad_center_on_neutral: bool = False,
                 gamepad_activation_samples: int = 3,
                 gamepad_center_delay: float = 1.0,
                 auto_reset_multiplier: float = 0.0,
                 # Return threshold - velocity must drop below this to re-enable detection
                 vel_return: float = 20.0,
                 # Deadzone - minimum velocity to consider as movement (filters noise)
                 deadzone_x: float = 5.0,
                 deadzone_y: float = 5.0,
                 deadzone_z: float = 5.0,
                 # Scale factor to convert raw values to deg/s
                 scale_factor: float = 0.25,
                 # Calibration duration
                 calibration_duration: float = 2.0,
                 # Smoothing window for velocity (number of samples)
                 smoothing_window: int = 5,
                 # Enable automatic drift correction (slowly adjusts baseline when stationary)
                 enable_drift_correction: bool = False,
                 # Axis inversions (flip polarity if directions are reversed)
                 invert_x: bool = False,
                 invert_y: bool = False,
                 invert_z: bool = False):
        
        self.vel_forward = vel_forward
        self.vel_backward = vel_backward
        self.vel_left = vel_left
        self.vel_right = vel_right
        self.z_left_threshold = z_left_threshold
        self.z_right_threshold = z_right_threshold
        self.use_z_for_lr = use_z_for_lr
        self.y_like_lr = y_like_lr
        self.gamepad_mode = gamepad_mode
        self.gamepad_lr_only = gamepad_lr_only
        self.gamepad_ud_only = gamepad_ud_only
        self.gamepad_center_on_neutral = gamepad_center_on_neutral
        self.gamepad_activation_samples = max(1, int(gamepad_activation_samples))
        self.gamepad_center_delay = max(0.0, float(gamepad_center_delay))
        self.auto_reset_multiplier = max(0.0, float(auto_reset_multiplier))
        self.vel_return = vel_return
        self.deadzone_x = deadzone_x
        self.deadzone_y = deadzone_y
        self.deadzone_z = deadzone_z
        self.scale_factor = scale_factor
        self.calibration_duration = calibration_duration
        self.smoothing_window = smoothing_window
        self.drift_correction_enabled = enable_drift_correction
        self.invert_x = invert_x
        self.invert_y = invert_y
        self.invert_z = invert_z
        
        # Saturation detection - ignore corrupted samples from LSL stream
        # Typical gyro raw values are -10000 to +10000, saturation is > 30000
        self.saturation_threshold = 30000.0  # If any axis exceeds this (raw), sample is corrupted
        self.last_valid_sample = None
        self.saturation_count = 0
        self.sample_debug_count = 0  # For initial debugging
        
        # Baseline (average at rest) - subtract from raw values
        self.baseline = np.array([0.0, 0.0, 0.0])
        self.baseline_initialized = False
        
        # Calibration
        self.is_calibrating = False
        self.calibration_samples = []
        
        # Smoothing buffer for velocity
        self.velocity_buffer = deque(maxlen=smoothing_window)
        
        # Drift correction buffer - tracks recent low-velocity samples to update baseline
        self.drift_correction_buffer = deque(maxlen=100)
        self.drift_update_threshold = 10.0  # Only update baseline if velocity < this (deg/s)
        
        # Stationary detection - auto re-zero when no movement detected
        self.stationary_threshold = 2.0  # deg/s - if all axes below this, consider stationary
        self.stationary_buffer = deque(maxlen=50)  # Track recent velocities to detect stillness
        self.is_stationary = False
        self.stationary_count = 0
        self.stationary_required = 25  # Number of consecutive stationary samples needed
        
        # State tracking - which directions are currently "locked out"
        # Once triggered, direction stays locked until velocity returns to baseline
        self.direction_active = {
            'forward': False,
            'backward': False,
            'left': False,
            'right': False
        }
        self.current_direction = 'center'
        self.gamepad_wait_neutral_after_center = False
        self.gamepad_center_block_until = 0.0
        self.gamepad_candidate_direction: Optional[str] = None
        self.gamepad_candidate_count = 0

    def _gamepad_scores(self, vel_x: float, vel_y: float, vel_z: float) -> dict:
        """Compute positive activation scores per direction; larger means more dominant."""
        lr_axis = vel_z if self.use_z_for_lr else vel_x
        left_thr = self.z_left_threshold if self.use_z_for_lr else self.vel_left
        right_thr = self.z_right_threshold if self.use_z_for_lr else self.vel_right
        if self.y_like_lr:
            # Mirror Z left/right trigger levels onto Y forward/backward.
            y_forward_thr = left_thr
            y_backward_thr = right_thr
        else:
            y_forward_thr = self.vel_forward
            y_backward_thr = self.vel_backward
        scores = {
            'left': max(0.0, lr_axis - left_thr),
            'right': max(0.0, -lr_axis - right_thr),
            'forward': 0.0,
            'backward': 0.0,
        }
        if self.gamepad_ud_only:
            scores['left'] = 0.0
            scores['right'] = 0.0
        if not self.gamepad_lr_only:
            scores['forward'] = max(0.0, vel_y - y_forward_thr)
            scores['backward'] = max(0.0, -vel_y - y_backward_thr)
        return scores

    def _gamepad_centered(self, vel_x: float, vel_y: float, vel_z: float) -> bool:
        """Return whether active direction has relaxed back to neutral zone."""
        lr_axis = vel_z if self.use_z_for_lr else vel_x
        if self.current_direction in ('left', 'right'):
            # For LR, use axis deadzone as center threshold.
            # Using vel_return here can be too large and cause immediate de-latch.
            lr_deadzone = self.deadzone_z if self.use_z_for_lr else self.deadzone_x
            return abs(lr_axis) < lr_deadzone
        if self.current_direction in ('forward', 'backward'):
            if self.y_like_lr:
                # Mirror Z neutral threshold onto Y when Y is configured like LR.
                return abs(vel_y) < (self.deadzone_z if self.use_z_for_lr else self.deadzone_y)
            return abs(vel_y) < self.vel_return
        return True

    def _gamepad_candidate_ready(self, direction: str) -> bool:
        if self.gamepad_candidate_direction == direction:
            self.gamepad_candidate_count += 1
        else:
            self.gamepad_candidate_direction = direction
            self.gamepad_candidate_count = 1
        return self.gamepad_candidate_count >= self.gamepad_activation_samples

    def _clear_gamepad_candidate(self) -> None:
        self.gamepad_candidate_direction = None
        self.gamepad_candidate_count = 0

    def _process_gamepad_mode(self, vel_x: float, vel_y: float, vel_z: float) -> Optional[str]:
        """Single-direction gamepad behavior with debounced opposite-direction cancel."""
        now = time.time()
        scores = self._gamepad_scores(vel_x, vel_y, vel_z)
        best_direction, best_score = max(scores.items(), key=lambda kv: kv[1])

        opposite = {
            'left': 'right',
            'right': 'left',
            'forward': 'backward',
            'backward': 'forward',
        }

        # If currently active, keep it latched until neutral or a debounced opposite motion.
        if self.current_direction != 'center':
            opp = opposite.get(self.current_direction)
            if self.gamepad_center_on_neutral and self._gamepad_centered(vel_x, vel_y, vel_z):
                previous_direction = self.current_direction
                self.current_direction = 'center'
                self._clear_gamepad_candidate()
                if previous_direction in ('forward', 'backward'):
                    self.gamepad_center_block_until = now + self.gamepad_center_delay
                return 'center'
            if opp and scores.get(opp, 0.0) > 0.0:
                if self._gamepad_candidate_ready(opp):
                    previous_direction = self.current_direction
                    self.current_direction = 'center'
                    self._clear_gamepad_candidate()
                    if previous_direction in ('forward', 'backward'):
                        self.gamepad_center_block_until = now + self.gamepad_center_delay
                    self.gamepad_wait_neutral_after_center = True
                    return 'center'
                return None
            self._clear_gamepad_candidate()
            return None

        # Optional delay gate after forward/backward returns to center.
        if now < self.gamepad_center_block_until:
            return None

        if self.gamepad_wait_neutral_after_center:
            if max(scores.values()) <= 0.0:
                self.gamepad_wait_neutral_after_center = False
            self._clear_gamepad_candidate()
            return None

        # From center, choose exactly one dominant direction if any threshold is exceeded.
        if best_score > 0.0:
            if self._gamepad_candidate_ready(best_direction):
                self.current_direction = best_direction
                self._clear_gamepad_candidate()
                return best_direction
            return None

        self._clear_gamepad_candidate()
        return None
        
    def start_calibration(self):
        """Begin calibration - device should be stationary."""
        self.is_calibrating = True
        self.calibration_samples = []
        # RESET baseline to zero - this ensures recalibration works properly
        self.baseline = np.array([0.0, 0.0, 0.0])
        self.baseline_initialized = False
        # Clear all buffers to start fresh
        self.velocity_buffer.clear()
        self.drift_correction_buffer.clear()
        self.stationary_buffer.clear()
        # Reset stationary detection
        self.stationary_count = 0
        self.is_stationary = False
        self.current_direction = 'center'
        self.gamepad_wait_neutral_after_center = False
        self.gamepad_center_block_until = 0.0
        self.gamepad_candidate_direction = None
        self.gamepad_candidate_count = 0
        # Reset all active states
        for key in self.direction_active:
            self.direction_active[key] = False
        print("\n" + "="*60)
        print("CALIBRATION STARTED")
        print(f"   Keep device STILL at desired ZERO position for {self.calibration_duration} seconds...")
        print("="*60)
        
    def add_calibration_sample(self, gyro_sample: np.ndarray):
        """Add sample during calibration."""
        # Apply scale factor first, then store for baseline calculation
        scaled_sample = gyro_sample * self.scale_factor
        self.calibration_samples.append(scaled_sample.copy())
        
    def finish_calibration(self, sfreq: float):
        """Complete calibration and compute baseline."""
        if len(self.calibration_samples) >= int(self.calibration_duration * sfreq * 0.8):
            self.baseline = np.mean(self.calibration_samples, axis=0)
            self.baseline_initialized = True
            self.is_calibrating = False
            
            # Show some raw samples for debugging
            if len(self.calibration_samples) > 0:
                sample_raw = self.calibration_samples[0] / self.scale_factor  # Reverse scale to show raw
                print(f"\n  DEBUG - Sample raw value: X={sample_raw[0]:.2f}, Y={sample_raw[1]:.2f}, Z={sample_raw[2]:.2f}")
            
            print("\n" + "="*60)
            print("CALIBRATION COMPLETE - ZERO REFERENCE SET")
            print(f"  Baseline (scaled): X={self.baseline[0]:.2f}, Y={self.baseline[1]:.2f}, Z={self.baseline[2]:.2f}")
            print(f"\n  Current position = ZERO reference")
            print(f"  All velocities measured as DIFFERENCE from this point")
            print(f"\n  Velocity Thresholds (deg/s):")
            print(f"    Forward:  {self.vel_forward:>6.1f}")
            print(f"    Backward: {self.vel_backward:>6.1f}")
            print(f"    Left:     {self.vel_left:>6.1f}")
            print(f"    Right:    {self.vel_right:>6.1f}")
            print(f"    Return:   {self.vel_return:>6.1f}")
            print(f"\n  Deadzones (deg/s):")
            print(f"    X-axis:   {self.deadzone_x:>6.1f}")
            print(f"    Y-axis:   {self.deadzone_y:>6.1f}")
            print(f"    Z-axis:   {self.deadzone_z:>6.1f}")
            print(f"\n  Scale Factor: {self.scale_factor}")
            print(f"  Drift Correction: {'ENABLED' if self.drift_correction_enabled else 'DISABLED'}")
            print("="*60)
            print("Ready! Tilt your head to send commands.")
            print("Press 'R' to recalibrate to NEW zero position, 'Q' to quit.\n")
            return True
        else:
            print("⚠ Not enough calibration samples, retrying...")
            self.is_calibrating = False
            return False
            
    def process_sample(self, gyro_sample: np.ndarray) -> Tuple[Optional[str], np.ndarray]:
        """
        Process gyro sample and detect direction.
        
        Args:
            gyro_sample: Raw 3-axis gyro data [X, Y, Z]
            
        Returns:
            (direction, corrected_velocity) where direction is 'forward', 'backward', 'left', 'right', or None
        """
        # DEBUG: Show first 5 raw samples to verify channels are correct
        if self.sample_debug_count < 5:
            print(f"DEBUG Raw sample #{self.sample_debug_count + 1}: X={gyro_sample[0]:.2f}, Y={gyro_sample[1]:.2f}, Z={gyro_sample[2]:.2f}")
            self.sample_debug_count += 1
            if self.sample_debug_count == 5:
                print("(Debug output complete - should see small values if gyro channels correct)\n")
        
        # SATURATION CHECK: Detect corrupted samples from LSL stream
        # When EEG channels saturate, they corrupt the gyro channels too
        if np.any(np.abs(gyro_sample) > self.saturation_threshold):
            self.saturation_count += 1
            if self.saturation_count == 1:
                print(f"\n⚠️  SATURATION DETECTED - Ignoring corrupted samples...")
                print(f"    Raw values: X={gyro_sample[0]:.1f}, Y={gyro_sample[1]:.1f}, Z={gyro_sample[2]:.1f}")
            # Return last valid velocity (or zero if none)
            if self.last_valid_sample is not None:
                return None, self.last_valid_sample
            else:
                return None, np.array([0.0, 0.0, 0.0])
        else:
            # Valid sample - reset saturation counter
            if self.saturation_count > 0:
                print(f"✓ Saturation cleared after {self.saturation_count} corrupted samples\n")
                self.saturation_count = 0
        
        # Apply scale factor first, then subtract baseline (baseline is in scaled units)
        scaled = gyro_sample * self.scale_factor
        corrected = scaled - self.baseline
        
        # Add to smoothing buffer
        self.velocity_buffer.append(corrected.copy())
        
        # Compute smoothed velocity (average over buffer)
        if len(self.velocity_buffer) >= self.smoothing_window:
            smoothed_vel = np.mean(list(self.velocity_buffer), axis=0)
        else:
            smoothed_vel = corrected
        
        # STATIONARY DETECTION: Check if device has stopped moving
        # If no movement detected, automatically update baseline to current position (auto re-zero)
        vel_magnitude = np.linalg.norm(smoothed_vel)
        self.stationary_buffer.append(vel_magnitude)
        
        # Check if device is stationary (all recent velocities are low)
        if len(self.stationary_buffer) >= 10:
            recent_velocities = list(self.stationary_buffer)[-10:]
            max_recent_vel = max(recent_velocities)
            
            if max_recent_vel < self.stationary_threshold:
                self.stationary_count += 1
                
                # If stationary for enough samples, update baseline to current position
                if self.stationary_count >= self.stationary_required:
                    if not self.is_stationary:
                        # Just became stationary - update baseline to lock in new position
                        self.baseline = scaled.copy()
                        self.is_stationary = True
                        # Recalculate corrected velocity with new baseline
                        corrected = scaled - self.baseline
                        # Clear buffers
                        self.velocity_buffer.clear()
                        self.velocity_buffer.append(corrected.copy())
                        smoothed_vel = corrected
            else:
                # Movement detected
                self.stationary_count = 0
                self.is_stationary = False
        
        # Continuous drift correction: if velocity is very low (device likely stationary),
        # slowly update baseline to compensate for sensor drift
        if self.drift_correction_enabled:
            if vel_magnitude < self.drift_update_threshold:
                # Device is stationary - track the raw scaled value for baseline update
                self.drift_correction_buffer.append(scaled.copy())
                
                # Update baseline every 50 samples when stationary
                if len(self.drift_correction_buffer) >= 50:
                    # Use exponential moving average to slowly adjust baseline
                    recent_baseline = np.mean(list(self.drift_correction_buffer), axis=0)
                    alpha = 0.05  # Slow adaptation rate
                    self.baseline = self.baseline * (1 - alpha) + recent_baseline * alpha
            
        # Extract axis velocities
        # AXIS MAPPING for Smarting24 (worn on back of head):
        vel_x = smoothed_vel[0]  # X-axis: Roll (left/right tilt)
        vel_y = smoothed_vel[1]  # Y-axis: Pitch (forward/backward tilt)
        vel_z = smoothed_vel[2]  # Z-axis: Yaw (rotation) - NOT USED for WASD
        
        # Apply axis inversions if needed
        if self.invert_x:
            vel_x = -vel_x
        if self.invert_y:
            vel_y = -vel_y
        if self.invert_z:
            vel_z = -vel_z
        
        # Optional emergency reset for truly extreme velocity spikes. Disabled by
        # default because normal intentional movement can exceed low thresholds.
        if self.auto_reset_multiplier > 0.0:
            x_reset_threshold = max(self.vel_left, self.vel_right) * self.auto_reset_multiplier
            if x_reset_threshold > 0.0 and abs(vel_x) > x_reset_threshold:
                self.baseline[0] = scaled[0]
                vel_x = 0.0
                self.direction_active['left'] = False
                self.direction_active['right'] = False
                print(f"[AUTO-RESET X-axis: {scaled[0]:.2f}]")

            y_reset_threshold = max(self.vel_forward, self.vel_backward) * self.auto_reset_multiplier
            if y_reset_threshold > 0.0 and abs(vel_y) > y_reset_threshold:
                self.baseline[1] = scaled[1]
                vel_y = 0.0
                self.direction_active['forward'] = False
                self.direction_active['backward'] = False
                print(f"[AUTO-RESET Y-axis: {scaled[1]:.2f}]")
        
        # Apply deadzones - ignore velocities below noise threshold
        # ALSO: If within deadzone, immediately unlock ALL directions (instant return to rest)
        within_deadzone_x = abs(vel_x) < self.deadzone_x
        within_deadzone_y = abs(vel_y) < self.deadzone_y
        within_deadzone_z = abs(vel_z) < self.deadzone_z
        
        if within_deadzone_x:
            vel_x = 0.0
            # Instantly unlock left/right when returning to deadzone
            self.direction_active['left'] = False
            self.direction_active['right'] = False
            
        if within_deadzone_y:
            vel_y = 0.0
            # Instantly unlock forward/backward when returning to deadzone
            self.direction_active['forward'] = False
            self.direction_active['backward'] = False
            
        if within_deadzone_z:
            vel_z = 0.0
        
        if self.gamepad_mode:
            detected_direction = self._process_gamepad_mode(vel_x, vel_y, vel_z)
            display_vel = smoothed_vel
            if detected_direction is not None:
                # On state switch, zero only the reported velocity to the caller/UI.
                # Keep internal baseline/filter state untouched to avoid fake retriggers.
                display_vel = np.zeros_like(smoothed_vel)
            self.last_valid_sample = smoothed_vel.copy()
            return detected_direction, display_vel

        detected_direction = None
        
        # Check each direction and update lock state
        
        # FORWARD: Y-axis positive velocity (head tilts forward)
        if vel_y > self.vel_forward:
            if not self.direction_active['forward']:
                detected_direction = 'forward'
                self.direction_active['forward'] = True
        elif abs(vel_y) < self.vel_return and not within_deadzone_y:
            # Unlock if dropped below return threshold (but not already unlocked by deadzone)
            self.direction_active['forward'] = False
            
        # BACKWARD: Y-axis negative velocity (head tilts backward)
        if vel_y < -self.vel_backward:
            if not self.direction_active['backward']:
                detected_direction = 'backward'
                self.direction_active['backward'] = True
        elif abs(vel_y) < self.vel_return and not within_deadzone_y:
            self.direction_active['backward'] = False
            
        if self.use_z_for_lr:
            # Z-axis positive triggers LEFT, Z-axis negative triggers RIGHT.
            if vel_z > self.z_left_threshold:
                if not self.direction_active['left']:
                    detected_direction = 'left'
                    self.direction_active['left'] = True
            elif abs(vel_z) < self.vel_return and not within_deadzone_z:
                self.direction_active['left'] = False

            if vel_z < -self.z_right_threshold:
                if not self.direction_active['right']:
                    detected_direction = 'right'
                    self.direction_active['right'] = True
            elif abs(vel_z) < self.vel_return and not within_deadzone_z:
                self.direction_active['right'] = False
        else:
            # LEFT: X-axis negative velocity (head tilts left)
            if vel_x < -self.vel_left:
                if not self.direction_active['left']:
                    detected_direction = 'left'
                    self.direction_active['left'] = True
            elif abs(vel_x) < self.vel_return and not within_deadzone_x:
                self.direction_active['left'] = False

            # RIGHT: X-axis positive velocity (head tilts right)
            if vel_x > self.vel_right:
                if not self.direction_active['right']:
                    detected_direction = 'right'
                    self.direction_active['right'] = True
            elif abs(vel_x) < self.vel_return and not within_deadzone_x:
                self.direction_active['right'] = False
        
        # Store last valid velocity for saturation handling
        self.last_valid_sample = smoothed_vel.copy()
        
        return detected_direction, smoothed_vel


def send_keypress(direction: str, key_map: dict):
    """Send a single keypress for the detected direction."""
    if not _key_output_available:
        return
        
    key = key_map.get(direction)
    if key:
        try:
            pydirectinput.press(key)
            print(f">>> {direction.upper():9s} detected -> Key '{key}' pressed")
        except Exception as e:
            print(f"Error sending key: {e}")


def parse_channel_indices(channels_arg: str) -> List[int]:
    """Parse comma-separated channel indices."""
    return [int(x.strip()) for x in channels_arg.split(',')]


def parse_key_mapping(mapping_arg: str) -> dict:
    """
    Parse key mapping string.
    
    Format: "forward:w,backward:s,left:a,right:d"
    """
    mapping = {}
    for pair in mapping_arg.split(','):
        direction, key = pair.split(':')
        mapping[direction.strip()] = resolve_key_token(key.strip())
    return mapping


def resolve_key_token(token: str) -> str:
    """Convert mapping token to pydirectinput-compatible key name."""
    if not token:
        return token
    t = token.strip().lower()
    special = {
        'left': 'left',
        'right': 'right',
        'up': 'up',
        'down': 'down',
        'space': 'space',
        'enter': 'enter',
        'esc': 'esc',
        'tab': 'tab',
    }
    if t in special:
        return special[t]
    return token


def on_key_press(key, detector: VelocityDetector):
    """Keyboard listener callback."""
    try:
        # Handle 'r' or 'R' for recalibration
        if hasattr(key, 'char') and key.char and key.char.lower() == 'r':
            detector.start_calibration()
        # Handle 'q' or 'Q' to quit
        elif hasattr(key, 'char') and key.char and key.char.lower() == 'q':
            print("\n[Quit requested]")
            return False  # Stop listener
    except Exception:
        pass


class _TeeStream:
    """Mirror writes to both console and a log file."""

    def __init__(self, original_stream, log_stream):
        self._original_stream = original_stream
        self._log_stream = log_stream

    def write(self, data):
        self._original_stream.write(data)
        self._log_stream.write(data)

    def flush(self):
        self._original_stream.flush()
        self._log_stream.flush()

    def isatty(self):
        return self._original_stream.isatty()


def setup_output_logging(log_file: str):
    """Tee stdout/stderr to a log file when log_file is provided."""
    if not log_file:
        return None

    log_path = Path(log_file)
    if log_path.exists() and log_path.is_dir():
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = log_path / f'gyro_detector_{stamp}.log'

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, mode='a', encoding='utf-8', buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_handle)
    sys.stderr = _TeeStream(original_stderr, log_handle)

    header = f"\n{'=' * 70}\nLog started: {datetime.now().isoformat()}\n{'=' * 70}\n"
    print(header, end='')
    print(f"Logging script output to: {log_path}")

    def _restore_streams():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        try:
            log_handle.flush()
            log_handle.close()
        except Exception:
            pass

    atexit.register(_restore_streams)
    return log_handle


class OverlayStateWriter:
    """Writes gamepad state to JSON file for overlay display."""
    
    def __init__(self, state_file: str):
        self.state_file = resolve_state_file(state_file)
        self.last_write_time = 0.0
        self.write_interval = 0.05  # Throttle writes to max 20 updates/sec
    
    def update(
        self,
        command: str,
        active_states: dict,
        output_text: str = "idle",
        cooldown_until: float = 0.0,
        cooldown_seconds: float = 0.0,
        cooldown_source: str = "",
        force: bool = False,
    ):
        """Update overlay state file with current gamepad status."""
        if not self.state_file:
            return
        
        now = time.time()
        if not force and (now - self.last_write_time) < self.write_interval:
            return  # Throttle writes
        
        try:
            state_data = {
                "command": command or "center",
                "active_states": active_states,
                "output": output_text,
                "timestamp": now,
            }
            state_data.update(cooldown_payload(cooldown_until, cooldown_seconds, cooldown_source))
            
            # Atomic write: temp file then rename
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state_data, f)
            temp_file.replace(self.state_file)
            
            self.last_write_time = now
        except Exception:
            pass  # Silently fail if overlay file write fails


def main():
    ensure_admin_privileges()

    ap = argparse.ArgumentParser(description="Gyroscope velocity-based detector for Smarting24")
    
    # Stream parameters
    ap.add_argument('--sfreq', type=float, default=0.0, 
                    help='Sampling rate (Hz). If 0, auto-detect from stream')
    ap.add_argument('--gyro-channels', type=str, default='0,1,2',
                    help='Gyroscope channel indices as CSV (e.g., "0,1,2")')
    ap.add_argument('--stream-type', type=str, default='EEG',
                    help='LSL stream type to connect to (default: EEG)')
    
    # Velocity thresholds (deg/s)
    ap.add_argument('--vel-forward', type=float, default=120.0,
                    help='Velocity threshold for forward tilt (deg/s)')
    ap.add_argument('--vel-backward', type=float, default=120.0,
                    help='Velocity threshold for backward tilt (deg/s)')
    ap.add_argument('--vel-left', type=float, default=80.0,
                    help='Velocity threshold for left tilt (deg/s)')
    ap.add_argument('--vel-right', type=float, default=80.0,
                    help='Velocity threshold for right tilt (deg/s)')
    ap.add_argument('--z-left-threshold', type=float, default=80.0,
                    help='If --use-z-for-lr is set: Z+ threshold to trigger LEFT (deg/s)')
    ap.add_argument('--z-right-threshold', type=float, default=80.0,
                    help='If --use-z-for-lr is set: Z- threshold magnitude to trigger RIGHT (deg/s)')
    ap.add_argument('--use-z-for-lr', action='store_true',
                    help='Use Z axis instead of X axis for left/right detection')
    ap.add_argument('--y-like-lr', action='store_true',
                    help='Make Y forward/backward behave like left/right in gamepad mode (deadzone center + opposite cancel)')
    ap.add_argument('--gamepad-mode', action='store_true',
                    help='Gamepad mode: only one dominant direction can be active at a time')
    ap.add_argument('--gamepad-lr-only', action='store_true',
                    help='In gamepad mode, restrict movement detection to left/right only (disables up/down via Y axis)')
    ap.add_argument('--gamepad-ud-only', action='store_true',
                    help='In gamepad mode, restrict movement detection to up/down only (disables left/right)')
    ap.add_argument('--gamepad-center-on-neutral', action='store_true',
                    help='In gamepad mode, return to center when active axis comes back into neutral zone')
    ap.add_argument('--gamepad-activation-samples', type=int, default=3,
                    help='Consecutive samples required to activate a direction from center in gamepad mode')
    ap.add_argument('--gamepad-center-delay', type=float, default=1.0,
                    help='Seconds to wait in center after forward/backward returns to center before allowing any new direction')
    ap.add_argument('--auto-reset-multiplier', type=float, default=0.0,
                    help='Reset an axis baseline if velocity exceeds threshold * multiplier; <=0 disables this behavior')
    ap.add_argument('--vel-return', type=float, default=20.0,
                    help='Return threshold - velocity must drop below this to unlock (deg/s)')
    
    # Deadzone thresholds (noise filtering)
    ap.add_argument('--deadzone-x', type=float, default=5.0,
                    help='Deadzone for X-axis - ignores velocity below this (deg/s)')
    ap.add_argument('--deadzone-y', type=float, default=5.0,
                    help='Deadzone for Y-axis - ignores velocity below this (deg/s)')
    ap.add_argument('--deadzone-z', type=float, default=5.0,
                    help='Deadzone for Z-axis - ignores velocity below this (deg/s)')
    
    # Processing parameters
    ap.add_argument('--scale-factor', type=float, default=0.25,
                    help='Scale factor to convert raw values to deg/s')
    ap.add_argument('--calibration-duration', type=float, default=2.0,
                    help='Calibration duration in seconds')
    ap.add_argument('--smoothing-window', type=int, default=5,
                    help='Number of samples to average for velocity smoothing')
    ap.add_argument('--enable-drift-correction', action='store_true',
                    help='Enable automatic drift correction (adjusts baseline when stationary)')
    
    # Axis inversions
    ap.add_argument('--invert-x', action='store_true',
                    help='Invert X-axis polarity (flip left/right)')
    ap.add_argument('--invert-y', action='store_true',
                    help='Invert Y-axis polarity (flip forward/backward)')
    ap.add_argument('--invert-z', action='store_true',
                    help='Invert Z-axis polarity (flip rotation)')
    
    # Key mapping
    ap.add_argument('--key-mapping', type=str, default='forward:w,backward:s,left:a,right:d',
                    help='Key mapping (e.g., "forward:w,backward:s,left:a,right:d")')
    ap.add_argument('--output-keys', action='store_true',
                    help='Enable keyboard output (default: disabled, just prints)')
    ap.add_argument('--gamepad-repeat-interval', type=float, default=0.12,
                    help='In gamepad mode, seconds between repeated left/right keypresses while latched')
    ap.add_argument('--command-cooldown', type=float, default=2.0,
                    help='Seconds to block blink/MI commands after a non-center gyro direction; <=0 disables')
    
    # Debug options
    ap.add_argument('--verbose', action='store_true',
                    help='Print detailed velocity information')
    ap.add_argument('--log-file', type=str, default='',
                    help='Log file path for saving script output (console output is still shown)')
    ap.add_argument('--overlay-state-file', type=str, default='',
                    help='JSON file path for overlay state updates (e.g., gamepad_state.json)')
    
    args = ap.parse_args()
    log_handle = setup_output_logging(args.log_file)
    
    # Parse key mapping
    key_map = parse_key_mapping(args.key_mapping)
    
    # Parse gyro channel indices
    gyro_channels = parse_channel_indices(args.gyro_channels)
    if len(gyro_channels) != 3:
        print(f"Error: Expected 3 gyro channels, got {len(gyro_channels)}")
        return
    
    # Find LSL stream
    print(f"Looking for LSL stream (type='{args.stream_type}')...")
    streams = resolve_byprop('type', args.stream_type, timeout=5.0)
    if not streams:
        print(f"Error: No LSL stream of type '{args.stream_type}' found")
        return
    
    stream_info = streams[0]
    print(f"Found stream: {stream_info.name()}")
    print(f"  Channels: {stream_info.channel_count()}")
    print(f"  Sampling rate: {stream_info.nominal_srate()} Hz")
    
    # Get sampling rate
    sfreq = args.sfreq if args.sfreq > 0 else stream_info.nominal_srate()
    if sfreq <= 0:
        print("Error: Could not determine sampling rate")
        return
    print(f"  Using sfreq: {sfreq} Hz")
    
    # Create inlet
    inlet = StreamInlet(stream_info, max_buflen=int(sfreq * 2))
    
    # Create detector
    detector = VelocityDetector(
        vel_forward=args.vel_forward,
        vel_backward=args.vel_backward,
        vel_left=args.vel_left,
        vel_right=args.vel_right,
        z_left_threshold=args.z_left_threshold,
        z_right_threshold=args.z_right_threshold,
        use_z_for_lr=args.use_z_for_lr,
        y_like_lr=args.y_like_lr,
        gamepad_mode=args.gamepad_mode,
        gamepad_lr_only=args.gamepad_lr_only,
        gamepad_ud_only=args.gamepad_ud_only,
        gamepad_center_on_neutral=args.gamepad_center_on_neutral,
        gamepad_activation_samples=args.gamepad_activation_samples,
        gamepad_center_delay=args.gamepad_center_delay,
        auto_reset_multiplier=args.auto_reset_multiplier,
        vel_return=args.vel_return,
        deadzone_x=args.deadzone_x,
        deadzone_y=args.deadzone_y,
        deadzone_z=args.deadzone_z,
        scale_factor=args.scale_factor,
        calibration_duration=args.calibration_duration,
        smoothing_window=args.smoothing_window,
        enable_drift_correction=args.enable_drift_correction,
        invert_x=args.invert_x,
        invert_y=args.invert_y,
        invert_z=args.invert_z
    )
    
    # Start calibration
    detector.start_calibration()
    
    # Start keyboard listener if available
    listener = None
    if _listener_available:
        listener = Listener(on_press=lambda key: on_key_press(key, detector))
        listener.start()
        print("Keyboard listener active (press 'R' to recalibrate, 'Q' to quit)")
    else:
        print("Keyboard listener unavailable (pynput not installed)")
    
    print("\nStarting detection loop...\n")
    
    # Initialize overlay state writer
    overlay_writer = OverlayStateWriter(args.overlay_state_file) if args.overlay_state_file else None
    if overlay_writer and overlay_writer.state_file:
        print(f"Overlay state will be written to: {overlay_writer.state_file}\n")
    
    try:
        sample_count = 0
        command_cooldown_seconds = max(0.0, float(args.command_cooldown))
        command_cooldown_until = 0.0
        command_cooldown_source = ""
        last_repeat_time = {
            'left': 0.0,
            'right': 0.0,
            'forward': 0.0,
            'backward': 0.0,
        }
        while True:
            # Pull sample from LSL
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample is None:
                continue
            
            sample_count += 1
            
            # Extract gyro channels
            try:
                gyro_sample = np.array([sample[ch] for ch in gyro_channels])
            except IndexError:
                print(f"Error: Channel indices {gyro_channels} out of range (stream has {len(sample)} channels)")
                break
            
            # Handle calibration
            if detector.is_calibrating:
                detector.add_calibration_sample(gyro_sample)
                if len(detector.calibration_samples) >= int(detector.calibration_duration * sfreq):
                    detector.finish_calibration(sfreq)
                continue
            
            # Process sample
            direction, velocity = detector.process_sample(gyro_sample)
            output_state_text = "idle"
            cooldown_started = False

            if direction in ('left', 'right', 'forward', 'backward') and command_cooldown_seconds > 0.0:
                now = time.time()
                command_cooldown_until = now + command_cooldown_seconds
                command_cooldown_source = direction
                cooldown_started = True

            # In gamepad mode, keep repeating left/right until center is reached.
            if args.gamepad_mode and args.output_keys and _key_output_available:
                active_dir = detector.current_direction
                if active_dir in ('left', 'right', 'forward', 'backward'):
                    now = time.time()
                    interval = max(0.01, args.gamepad_repeat_interval)
                    if (now - last_repeat_time[active_dir]) >= interval:
                        try:
                            repeat_key_map = {
                                'left': 'left',
                                'right': 'right',
                                # Reversed as requested: forward->down, backward->up.
                                'forward': 'down',
                                'backward': 'up',
                            }
                            repeat_key = repeat_key_map[active_dir]
                            pydirectinput.press(repeat_key)
                            output_state_text = f"holding {active_dir}"
                            last_repeat_time[active_dir] = now
                            if args.verbose:
                                label = {
                                    'left': 'LEFT',
                                    'right': 'RIGHT',
                                    'forward': 'DOWN',
                                    'backward': 'UP',
                                }[active_dir]
                                print(f">>> {label:9s} repeat -> Arrow key pressed")
                        except Exception as e:
                            print(f"Error sending repeated arrow key: {e}")
            
            # Update overlay with current state
            if overlay_writer:
                current_cmd = detector.current_direction if args.gamepad_mode else (direction if direction else "center")
                if output_state_text == "idle" and direction:
                    output_state_text = f"detected {direction}"
                cooldown_remaining = max(0.0, command_cooldown_until - time.time())
                if output_state_text == "idle" and cooldown_remaining > 0.0:
                    output_state_text = f"gyro cooldown {cooldown_remaining:.1f}s"
                overlay_writer.update(
                    command=current_cmd,
                    active_states=detector.direction_active,
                    output_text=output_state_text,
                    cooldown_until=command_cooldown_until,
                    cooldown_seconds=command_cooldown_seconds,
                    cooldown_source=command_cooldown_source,
                    force=cooldown_started,
                )
            
            # Print verbose info
            if args.verbose and sample_count % 50 == 0:
                extra_state = f" | GamepadState={detector.current_direction}" if args.gamepad_mode else ""
                print(f"Velocity: X={velocity[0]:>7.2f}, Y={velocity[1]:>7.2f}, Z={velocity[2]:>7.2f}  |  "
                    f"Active: {[k for k, v in detector.direction_active.items() if v]}{extra_state}")
            
            # Handle detected direction
            if direction is not None:
                if direction == 'center':
                    print('>>> CENTER')
                    continue
                if cooldown_started:
                    print(f">>> Gyro command cooldown started: {command_cooldown_seconds:.1f}s after {direction}")
                if args.output_keys:
                    if args.gamepad_mode and direction in ('left', 'right', 'forward', 'backward'):
                        # In gamepad mode, key output is handled by the repeat loop only.
                        # This avoids a double-tap on first entry into a direction.
                        if args.verbose:
                            label = {
                                'left': 'LEFT',
                                'right': 'RIGHT',
                                'forward': 'UP',
                                'backward': 'DOWN',
                            }[direction]
                            print(f">>> {label:9s} detected -> LATCHED")
                    else:
                        send_keypress(direction, key_map)
                else:
                    key = key_map.get(direction, '?')
                    print(f">>> {direction.upper():9s} detected (would send '{key}' key)")
                    
    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    finally:
        if listener is not None:
            try:
                listener.stop()
            except:
                pass
        if log_handle is not None:
            try:
                log_handle.flush()
            except Exception:
                pass
        print("Cleanup complete.")


if __name__ == '__main__':
    main()
