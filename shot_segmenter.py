"""
shot_segmenter.py - With junk shot filtering
Added validate_shot() to reject physically impossible arcs.

Iteration 4 fix:
  - Angle validation now handles both left-to-right AND right-to-left shots
  - Uses absolute angle so direction doesn't matter
"""

import numpy as np
from collections import deque
from typing import Optional, Tuple, List


class ShotState:
    IDLE       = "IDLE"
    ASCENDING  = "ASCENDING"
    DESCENDING = "DESCENDING"
    COMPLETE   = "COMPLETE"


class ShotSegmenter:
    def __init__(
        self,
        fps: float,
        min_shot_frames: int = 8,
        max_shot_frames: int = 120,
        min_upward_px: int = 40,
        velocity_window: int = 6,
        cooldown_frames: int = 30,
        min_vy_trigger: float = 1.5,
        min_vy_descend: float = 3.0,
        # Validation thresholds
        min_arc_height_px: int = 30,
        max_flight_time_sec: float = 8.0,
        min_valid_angle: float = 5.0,    # reject near-horizontal shots
        max_valid_angle: float = 89.0,   # reject near-vertical (bad tracking)
    ):
        self.fps               = fps
        self.min_shot_frames   = min_shot_frames
        self.max_shot_frames   = max_shot_frames
        self.min_upward_px     = min_upward_px
        self.vel_window        = velocity_window
        self.cooldown_frames   = cooldown_frames
        self.min_vy_trigger    = min_vy_trigger
        self.min_vy_descend    = min_vy_descend
        self.min_arc_height    = min_arc_height_px
        self.max_flight_time   = max_flight_time_sec
        self.min_valid_angle   = min_valid_angle
        self.max_valid_angle   = max_valid_angle

        self.state = ShotState.IDLE
        self._shot_points: List[Tuple[int, int]] = []
        self._shot_frames: List[int] = []
        self._recent_positions: deque = deque(maxlen=velocity_window)
        self._peak_y: Optional[int] = None
        self._start_y: Optional[int] = None
        self._frames_in_state: int = 0
        self._cooldown: int = 0
        self.completed_shots: List[dict] = []

    def update(self, ball_pos, frame_idx):
        if self._cooldown > 0:
            self._cooldown -= 1

        if ball_pos is not None:
            self._recent_positions.append(ball_pos)

        vy = self._get_vertical_velocity()

        if self.state == ShotState.IDLE:
            if self._cooldown > 0:
                return self.state
            if ball_pos is not None and vy is not None and vy < -self.min_vy_trigger:
                self.state = ShotState.ASCENDING
                self._shot_points = [ball_pos]
                self._shot_frames = [frame_idx]
                self._start_y = ball_pos[1]
                self._peak_y  = ball_pos[1]
                self._frames_in_state = 0

        elif self.state == ShotState.ASCENDING:
            self._frames_in_state += 1
            if ball_pos is not None:
                self._shot_points.append(ball_pos)
                self._shot_frames.append(frame_idx)
                if ball_pos[1] < self._peak_y:
                    self._peak_y = ball_pos[1]
            if vy is not None and vy > self.min_vy_descend and self._frames_in_state > 8:
                self.state = ShotState.DESCENDING
                self._frames_in_state = 0
            if self._frames_in_state > self.max_shot_frames:
                self._reset()

        elif self.state == ShotState.DESCENDING:
            self._frames_in_state += 1
            if ball_pos is not None:
                self._shot_points.append(ball_pos)
                self._shot_frames.append(frame_idx)
            at_bottom = (ball_pos is not None and self._start_y is not None
                         and ball_pos[1] >= self._start_y)
            timed_out = self._frames_in_state > 60
            if at_bottom or timed_out:
                self._maybe_complete()

        elif self.state == ShotState.COMPLETE:
            self.state = ShotState.IDLE
            self._cooldown = self.cooldown_frames

        return self.state

    def validate_shot(self, metrics: dict) -> Tuple[bool, str]:
        """
        Returns (is_valid, reason_if_invalid).
        Uses absolute angle — valid for both left-to-right and right-to-left shots.
        """
        if "error" in metrics:
            return False, "not enough data"

        # Flight time check
        try:
            ft = float(metrics["flight_time_sec"].replace("s", "").strip())
            if ft > self.max_flight_time:
                return False, f"flight time too long ({ft:.1f}s)"
        except Exception:
            pass

        # Arc height check
        try:
            ah = float(metrics["arc_height_px"].replace("px", "").strip())
            if ah < self.min_arc_height:
                return False, f"arc too flat ({ah:.0f}px)"
        except Exception:
            pass

        # Release angle check — use ABSOLUTE value so direction doesn't matter
        try:
            angle = float(metrics["release_angle_deg"].replace("°", "").strip())
            abs_angle = abs(angle)
            if abs_angle < self.min_valid_angle:
                return False, f"angle too flat ({angle:.1f}°)"
            if abs_angle > self.max_valid_angle:
                return False, f"angle too steep ({angle:.1f}°)"
        except Exception:
            pass

        return True, "ok"

    def _maybe_complete(self):
        n = len(self._shot_points)
        if n >= self.min_shot_frames:
            ys = [p[1] for p in self._shot_points]
            if (max(ys) - min(ys)) >= self.min_upward_px:
                self.completed_shots.append({
                    "points": list(self._shot_points),
                    "frames": list(self._shot_frames),
                    "peak_y": self._peak_y,
                    "start_y": self._start_y,
                })
                self.state = ShotState.COMPLETE
                self._reset_buffer()
                return
        self.state = ShotState.IDLE
        self._cooldown = 10
        self._reset_buffer()

    def _reset(self):
        self.state = ShotState.IDLE
        self._cooldown = 10
        self._reset_buffer()

    def _reset_buffer(self):
        self._shot_points = []
        self._shot_frames = []
        self._peak_y = None
        self._start_y = None
        self._frames_in_state = 0

    def _get_vertical_velocity(self):
        if len(self._recent_positions) < 2:
            return None
        ys = [p[1] for p in self._recent_positions]
        diffs = [ys[i+1] - ys[i] for i in range(len(ys) - 1)]
        return float(np.mean(diffs))

    def get_active_points(self):
        return list(self._shot_points)

    def get_latest_shot(self):
        return self.completed_shots[-1] if self.completed_shots else None