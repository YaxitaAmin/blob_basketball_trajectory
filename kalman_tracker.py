"""
kalman_tracker.py
Kalman Filter for smooth ball tracking and trajectory prediction during occlusion.

State vector: [x, y, vx, vy]
  - (x, y):   ball position in pixels
  - (vx, vy): velocity in pixels/frame

This allows the tracker to predict ball position even when detection fails.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class KalmanTracker:
    """
    Wraps OpenCV's KalmanFilter for basketball tracking.

    Usage:
        tracker = KalmanTracker()
        tracker.update((cx, cy))       # when ball is detected
        pos = tracker.predict()        # when ball is NOT detected (occlusion)
    """

    def __init__(self):
        # 4 state vars [x, y, vx, vy], 2 measurement vars [x, y]
        self.kf = cv2.KalmanFilter(4, 2)

        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix — we observe (x, y) only
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        # Process noise covariance (how much we trust the motion model)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2

        # Measurement noise covariance (how much we trust detections)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1

        # Initial error covariance
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.initialized = False
        self._consecutive_misses = 0
        self.MAX_MISSES = 8  # stop predicting after 8 frames of no detection

    def update(self, center: Tuple[int, int]):
        """Called when the ball is detected."""
        measurement = np.array([[np.float32(center[0])],
                                 [np.float32(center[1])]])

        if not self.initialized:
            self.kf.statePre = np.array(
                [[center[0]], [center[1]], [0.], [0.]], dtype=np.float32
            )
            self.kf.statePost = self.kf.statePre.copy()
            self.initialized = True
        else:
            self.kf.predict()
            self.kf.correct(measurement)

        self._consecutive_misses = 0

    def predict(self) -> Optional[Tuple[int, int]]:
        """Called when the ball is NOT detected (occlusion/blur)."""
        if not self.initialized:
            return None

        self._consecutive_misses += 1

        if self._consecutive_misses > self.MAX_MISSES:
            return None

        predicted = self.kf.predict()
        px = int(predicted[0][0])
        py = int(predicted[1][0])
        return (px, py)

    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Returns current estimated velocity (vx, vy) in pixels/frame."""
        if not self.initialized:
            return None
        vx = float(self.kf.statePost[2][0])
        vy = float(self.kf.statePost[3][0])
        return (vx, vy)

    def reset(self):
        """Reset tracker state (e.g., between shots)."""
        self.initialized = False
        self._consecutive_misses = 0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)