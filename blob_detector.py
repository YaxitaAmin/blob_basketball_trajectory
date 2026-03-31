"""
blob_detector.py
Detects the basketball using:
  - Primary:  Contour-based detection on HSV mask (more robust than SimpleBlobDetector)
  - Fallback: Shi-Tomasi corner detection when blob fails (motion blur / occlusion)

Iteration 2-3 fixes:
  - Single-ball constraint: picks BEST candidate per frame using a score
  - Motion continuity filter: rejects candidates too far from last known position
  - Circularity scored via contour area vs bounding circle area

Iteration 4 fixes:
  - ROI masking: ignores ceiling (top 15%) and bleachers/floor (bottom 40%)
  - Tighter saturation gate to reject gold bleacher seats
  - Corner fallback also respects ROI
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class BlobDetector:
    def __init__(
        self,
        min_radius_px: int = 8,
        max_radius_px: int = 55,
        min_circularity: float = 0.33,
        use_corner_fallback: bool = True,
        max_jump_px: int = 180,
        roi_top: float = 0.15,     # ignore top X% of frame (ceiling fixtures)
        roi_bottom: float = 0.75   # was 0.60,  # ignore below Y% of frame (bleachers/floor)
    ):
        self.min_radius  = min_radius_px
        self.max_radius  = max_radius_px
        self.min_circ    = min_circularity
        self.use_corner_fallback = use_corner_fallback
        self.max_jump_px = max_jump_px
        self.roi_top     = roi_top
        self.roi_bottom  = roi_bottom
        self._last_pos: Optional[Tuple[int, int]] = None

    def detect(self, mask, frame):
        # Apply ROI before any detection
        mask = self._apply_roi(mask)
        center, radius = self._contour_detect(mask)
        if center is None and self.use_corner_fallback:
            center, radius = self._corner_fallback(mask, frame)
        if center is not None:
            self._last_pos = center
        return center, radius

    def reset(self):
        self._last_pos = None

    def _apply_roi(self, mask):
        """Zero out ceiling and bleacher/floor regions — ball only flies in the middle band."""
        mask = mask.copy()
        h = mask.shape[0]
        mask[:int(h * self.roi_top), :]    = 0  # blackout ceiling
        mask[int(h * self.roi_bottom):, :] = 0  # blackout bleachers + floor
        return mask

    def _contour_detect(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            radius = float(radius)

            if not (self.min_radius <= radius <= self.max_radius):
                continue

            circle_area = np.pi * radius ** 2
            circularity = area / circle_area  # 1.0 = perfect circle

            if circularity < self.min_circ:
                continue

            center = (int(cx), int(cy))

            # Motion continuity: skip blobs too far from last detection
            if self._last_pos is not None:
                dist = np.hypot(cx - self._last_pos[0], cy - self._last_pos[1])
                if dist > self.max_jump_px:
                    continue

            # Score: weighted circularity + relative size
            score = circularity * 0.7 + (radius / self.max_radius) * 0.3
            candidates.append((score, center, int(radius)))

        if not candidates:
            return None, None

        best = max(candidates, key=lambda x: x[0])
        return best[1], best[2]

    def _corner_fallback(self, mask, frame):
        """Shi-Tomasi fallback — also respects the ROI mask."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

        corners = cv2.goodFeaturesToTrack(
            masked_gray, maxCorners=20, qualityLevel=0.3, minDistance=5, mask=mask,
        )
        if corners is None or len(corners) < 3:
            return None, None

        corners = corners.reshape(-1, 2)
        cx = int(np.mean(corners[:, 0]))
        cy = int(np.mean(corners[:, 1]))

        if self._last_pos is not None:
            dist = np.hypot(cx - self._last_pos[0], cy - self._last_pos[1])
            if dist > self.max_jump_px:
                return None, None

        dists = np.sqrt(((corners - np.array([cx, cy])) ** 2).sum(axis=1))
        r = int(np.mean(dists))
        r = max(self.min_radius, min(r, self.max_radius))
        return (cx, cy), r

    def hough_circles(self, mask):
        circles = cv2.HoughCircles(
            mask, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
            param1=50, param2=20,
            minRadius=self.min_radius, maxRadius=self.max_radius,
        )
        if circles is None:
            return None, None
        circles = np.uint16(np.around(circles[0]))
        best = max(circles, key=lambda c: c[2])
        return (int(best[0]), int(best[1])), int(best[2])