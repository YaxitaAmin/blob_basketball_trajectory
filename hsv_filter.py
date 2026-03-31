"""
hsv_filter.py - Tuned for indoor gym lighting (test.mp4)
Ball HSV profile: H=20-24, S=140-170, V=196-249
"""

import cv2
import numpy as np


class HSVFilter:
    def __init__(self, mode="adaptive", blur_ksize=5):
        self.mode = mode
        self.blur_ksize = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1

    def apply(self, frame):
        blurred = cv2.GaussianBlur(frame, (self.blur_ksize, self.blur_ksize), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # m1: core orange ball — tight S floor to reject gold bleacher seats (S~80-100)
        m1 = cv2.inRange(hsv, np.array([8,  120,  80]), np.array([28, 255, 255]))
        # m2: red-wrap (ball edge can appear reddish)
        m2 = cv2.inRange(hsv, np.array([170,  60,  40]), np.array([180, 255, 255]))
        # m3: fallback for motion-blurred / partially occluded ball
        m3 = cv2.inRange(hsv, np.array([5,   100,  60]), np.array([30,  255, 255]))
        mask = cv2.bitwise_or(m1, cv2.bitwise_or(m2, m3))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def tune(self, frame):
        def nothing(x): pass
        cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
        for name, val, mx in [("H_low", 8, 179), ("H_high", 28, 179),
                               ("S_low", 120, 255), ("S_high", 255, 255),
                               ("V_low", 80, 255),  ("V_high", 255, 255)]:
            cv2.createTrackbar(name, "HSV Tuner", val, mx, nothing)
        lower, upper = None, None
        while True:
            h = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower = np.array([cv2.getTrackbarPos("H_low",  "HSV Tuner"),
                               cv2.getTrackbarPos("S_low",  "HSV Tuner"),
                               cv2.getTrackbarPos("V_low",  "HSV Tuner")])
            upper = np.array([cv2.getTrackbarPos("H_high", "HSV Tuner"),
                               cv2.getTrackbarPos("S_high", "HSV Tuner"),
                               cv2.getTrackbarPos("V_high", "HSV Tuner")])
            cv2.imshow("HSV Tuner", cv2.inRange(h, lower, upper))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
        return lower, upper