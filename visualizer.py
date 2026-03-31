"""
visualizer.py - Cleaner trajectory drawing with parabola fit overlay
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple


class Visualizer:
    def __init__(self, frame_width: int, frame_height: int):
        self.w = frame_width
        self.h = frame_height
        self.COLOR_BALL_DETECTED  = (0, 200, 255)
        self.COLOR_BALL_PREDICTED = (255, 100, 0)
        self.COLOR_TRAJ_START     = (0, 255, 0)
        self.COLOR_TRAJ_END       = (0, 0, 255)
        self.COLOR_PARABOLA       = (0, 255, 255)
        self.COLOR_TEXT           = (255, 255, 255)

    def draw(self, frame, ball_pos, detected_radius, trajectory_points,
             mask=None, is_predicted=False, metrics=None):
        vis = frame.copy()

        if len(trajectory_points) > 1:
            self._draw_trajectory(vis, trajectory_points)
            if len(trajectory_points) >= 10:
                self._draw_parabola_fit(vis, trajectory_points)

        if ball_pos is not None:
            r = detected_radius if detected_radius else 18
            color = self.COLOR_BALL_PREDICTED if is_predicted else self.COLOR_BALL_DETECTED
            cv2.circle(vis, ball_pos, r, color, 2)
            cv2.circle(vis, ball_pos, 4, color, -1)
            label = "PRED" if is_predicted else "BALL"
            cv2.putText(vis, label, (ball_pos[0] + r + 4, ball_pos[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if mask is not None:
            self._draw_debug_mask(vis, mask)
        if metrics:
            self._draw_hud(vis, metrics)
        return vis

    def _draw_trajectory(self, frame, points):
        """Draw smoothed trajectory as thick gradient line (green → red)."""
        n = len(points)
        for i in range(1, n):
            alpha = i / n
            color = self._lerp_color(self.COLOR_TRAJ_START, self.COLOR_TRAJ_END, alpha)
            thickness = max(2, int(alpha * 4))
            cv2.line(frame, points[i-1], points[i], color, thickness, cv2.LINE_AA)

    def _draw_parabola_fit(self, frame, points):
        """
        Fit a parabola and draw it as a cyan curve.

        Basketball shots are mostly VERTICAL (ball goes high up) so we fit
        x = f(y) instead of y = f(x). This avoids a flat/straight line when
        the x-range is small but y-range is large.
        """
        pts = np.array(points, dtype=np.float32)
        xs, ys = pts[:, 0], pts[:, 1]

        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()

        try:
            if y_range > x_range:
                # Vertical shot — fit x as function of y
                coeffs = np.polyfit(ys, xs, deg=2)
                poly   = np.poly1d(coeffs)
                y_min, y_max = int(ys.min()), int(ys.max())
                if y_max <= y_min:
                    return
                prev_pt = None
                for y in range(y_min, y_max, 3):
                    x = int(poly(y))
                    pt = (x, y)
                    if prev_pt is not None and 0 <= x < self.w:
                        cv2.line(frame, prev_pt, pt, self.COLOR_PARABOLA, 2, cv2.LINE_AA)
                    prev_pt = pt
            else:
                # Horizontal shot — fit y as function of x (original approach)
                coeffs = np.polyfit(xs, ys, deg=2)
                poly   = np.poly1d(coeffs)
                x_min, x_max = int(xs.min()), int(xs.max())
                if x_max <= x_min:
                    return
                prev_pt = None
                for x in range(x_min, x_max, 2):
                    y = int(poly(x))
                    pt = (x, y)
                    if prev_pt is not None and 0 <= y < self.h:
                        cv2.line(frame, prev_pt, pt, self.COLOR_PARABOLA, 2, cv2.LINE_AA)
                    prev_pt = pt
        except Exception:
            pass

    def _lerp_color(self, c1, c2, t):
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

    def _draw_debug_mask(self, frame, mask):
        thumb_w, thumb_h = 320, 180
        thumb = cv2.resize(mask, (thumb_w, thumb_h))
        thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        x_off = self.w - thumb_w - 10
        y_off = 10
        frame[y_off:y_off+thumb_h, x_off:x_off+thumb_w] = thumb_bgr
        cv2.rectangle(frame, (x_off-1, y_off-1),
                      (x_off+thumb_w+1, y_off+thumb_h+1), (200, 200, 200), 1)
        cv2.putText(frame, "HSV mask", (x_off+4, y_off+thumb_h+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    def _draw_hud(self, frame, metrics):
        x, y = 10, 25
        line_h = 20
        bg_h = len(metrics) * line_h + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (260, bg_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        for key, val in metrics.items():
            label = f"{key.replace('_', ' ').title()}: {val}"
            cv2.putText(frame, label, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_TEXT, 1, cv2.LINE_AA)
            y += line_h