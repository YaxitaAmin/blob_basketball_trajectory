"""
background_subtractor.py
Uses MOG2 background subtraction to isolate MOVING objects only.
This kills static false positives like graffiti, walls, fences.

The combined mask = HSV mask AND motion mask
→ must be BOTH orange-ish AND moving to be a candidate
"""

import cv2
import numpy as np


class BackgroundSubtractor:
    """
    Wraps OpenCV MOG2 background subtractor.
    Learns the static background over first N frames,
    then returns foreground (moving) masks.
    """

    def __init__(
        self,
        history: int = 200,
        var_threshold: float = 40,
        detect_shadows: bool = False,
        min_area: int = 80,
    ):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows,
        )
        self.min_area = min_area
        self._frame_count = 0
        self.warmup_frames = 30

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Returns binary motion mask (white = moving region)."""
        fg_mask = self.fgbg.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(fg_mask)
        for cnt in contours:
            if cv2.contourArea(cnt) >= self.min_area:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

        self._frame_count += 1
        return clean_mask

    def is_warmed_up(self) -> bool:
        return self._frame_count >= self.warmup_frames

    def reset(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=40, detectShadows=False
        )
        self._frame_count = 0