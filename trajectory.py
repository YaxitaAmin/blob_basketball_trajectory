"""
trajectory.py - Smoother trajectories using median filter + outlier rejection
"""

import numpy as np
from typing import List, Optional, Tuple, Dict


class TrajectoryAnalyzer:
    def __init__(self, smoothing_window: int = 9, max_points: int = 500):
        self.smoothing_window = smoothing_window if smoothing_window % 2 == 1 else smoothing_window + 1
        self.max_points = max_points
        self._raw_points: List[Tuple[int, int]] = []
        self._frame_indices: List[int] = []

    def add_point(self, center: Tuple[int, int], frame_idx: int):
        if len(self._raw_points) >= self.max_points:
            self._raw_points.pop(0)
            self._frame_indices.pop(0)
        self._raw_points.append(center)
        self._frame_indices.append(frame_idx)

    def clear(self):
        self._raw_points.clear()
        self._frame_indices.clear()

    def get_smoothed_points(self) -> List[Tuple[int, int]]:
        if len(self._raw_points) < 3:
            return list(self._raw_points)

        pts = np.array(self._raw_points, dtype=np.float32)
        pts = self._reject_outliers(pts)

        if len(pts) < 3:
            return [(int(p[0]), int(p[1])) for p in pts]

        # Gaussian-weighted smoothing — center point weighted most
        half = self.smoothing_window // 2
        smoothed = []
        for i in range(len(pts)):
            lo = max(0, i - half)
            hi = min(len(pts), i + half + 1)
            window = pts[lo:hi]
            n = len(window)
            weights = np.exp(-0.5 * ((np.arange(n) - (i - lo)) / (half / 2 + 1e-5)) ** 2)
            weights /= weights.sum()
            smoothed_pt = (weights[:, None] * window).sum(axis=0)
            smoothed.append((int(smoothed_pt[0]), int(smoothed_pt[1])))

        return smoothed

    def _reject_outliers(self, pts: np.ndarray, threshold: float = 60.0) -> np.ndarray:
        """Remove points that jump too far from their neighbors."""
        if len(pts) < 3:
            return pts
        keep = [True] * len(pts)
        for i in range(1, len(pts) - 1):
            expected = (pts[i-1] + pts[i+1]) / 2
            if np.linalg.norm(pts[i] - expected) > threshold:
                keep[i] = False
        return pts[keep]

    def compute_metrics(self, fps: float) -> Dict[str, str]:
        if len(self._raw_points) < 5:
            return {"error": "Not enough trajectory data to compute metrics"}

        pts = np.array(self.get_smoothed_points(), dtype=np.float32)
        xs, ys = pts[:, 0], pts[:, 1]

        # Use first 25% of trajectory for release angle
        # avoids sampling near the flat peak which gives ~0° angles
        n_look = max(3, len(pts) // 4)
        dx = xs[n_look] - xs[0]
        dy = ys[n_look] - ys[0]

        # abs(dx) handles both left-to-right and right-to-left shots correctly
        release_angle_deg = abs(np.degrees(np.arctan2(-dy, abs(dx) + 1e-5)))

        arc_height_px         = float(ys[0] - np.min(ys))
        n_frames              = self._frame_indices[-1] - self._frame_indices[0]
        flight_time_sec       = n_frames / fps if fps > 0 else 0.0
        horiz_displacement_px = abs(float(xs[-1] - xs[0]))
        peak_frame_idx        = int(np.argmin(ys))

        return {
            "release_angle_deg":       f"{release_angle_deg:.1f}°",
            "arc_height_px":           f"{arc_height_px:.0f} px",
            "flight_time_sec":         f"{flight_time_sec:.2f} s",
            "horizontal_displacement": f"{horiz_displacement_px:.0f} px",
            "total_frames_tracked":    str(len(self._raw_points)),
            "peak_at_frame":           str(self._frame_indices[peak_frame_idx] if peak_frame_idx < len(self._frame_indices) else "N/A"),
        }

    def fit_parabola(self) -> Optional[np.ndarray]:
        if len(self._raw_points) < 5:
            return None
        pts = np.array(self._raw_points, dtype=np.float32)
        try:
            return np.polyfit(pts[:, 0], pts[:, 1], deg=2)
        except np.linalg.LinAlgError:
            return None

    def get_raw_points(self):
        return list(self._raw_points)