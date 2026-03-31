"""
Basketball Shot Trajectory Tracker - MSML640 - Yaxita Amin
Full pipeline with shot segmentation + validation.
"""

import cv2
import argparse
from hsv_filter import HSVFilter
from blob_detector import BlobDetector
from background_subtractor import BackgroundSubtractor
from kalman_tracker import KalmanTracker
from trajectory import TrajectoryAnalyzer
from shot_segmenter import ShotSegmenter, ShotState
from visualizer import Visualizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     type=str, required=True)
    parser.add_argument("--output",    type=str, default="output_tracked.mp4")
    parser.add_argument("--show",      action="store_true")
    parser.add_argument("--debug",     action="store_true")
    parser.add_argument("--no-bg-sub", action="store_true")
    return parser.parse_args()


def main():
    args  = parse_args()
    cap   = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {args.input}")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    hsv_filter = HSVFilter()
    bg_sub     = BackgroundSubtractor()
    blob_det   = BlobDetector()
    kalman     = KalmanTracker()
    trajectory = TrajectoryAnalyzer()
    segmenter  = ShotSegmenter(fps=fps)
    visualizer = Visualizer(width, height)

    use_bg_sub   = not args.no_bg_sub
    frame_idx    = 0
    shot_count   = 0
    valid_shots  = 0

    print(f"[INFO] {width}x{height} @ {fps:.1f} FPS | BG sub: {'ON' if use_bg_sub else 'OFF'}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv_mask = hsv_filter.apply(frame)
        if use_bg_sub:
            motion_mask   = bg_sub.apply(frame)
            combined_mask = cv2.bitwise_and(hsv_mask, motion_mask)
        else:
            combined_mask = hsv_mask

        detected_center, detected_radius = blob_det.detect(combined_mask, frame)

        is_predicted = False
        if detected_center is not None:
            kalman.update(detected_center)
            ball_pos = detected_center
        else:
            ball_pos     = kalman.predict()
            is_predicted = ball_pos is not None

        state = segmenter.update(ball_pos, frame_idx)

        if state in (ShotState.ASCENDING, ShotState.DESCENDING):
            if ball_pos is not None:
                trajectory.add_point(ball_pos, frame_idx)

        elif state == ShotState.COMPLETE:
            shot_count += 1
            metrics = trajectory.compute_metrics(fps)

            is_valid, reason = segmenter.validate_shot(metrics)
            if is_valid:
                valid_shots += 1
                print(f"\n✅ Shot #{valid_shots} (raw #{shot_count}) @ frame {frame_idx}")
                for k, v in metrics.items():
                    print(f"  {k}: {v}")
            else:
                print(f"  ❌ Shot #{shot_count} rejected: {reason}")

            trajectory.clear()
            kalman.reset()
            blob_det.reset()

        active_points = segmenter.get_active_points()
        debug_mask    = combined_mask if args.debug else None

        state_color = {
            ShotState.IDLE:       (128, 128, 128),
            ShotState.ASCENDING:  (0, 255, 0),
            ShotState.DESCENDING: (0, 165, 255),
            ShotState.COMPLETE:   (0, 255, 255),
        }.get(state, (255, 255, 255))

        vis = visualizer.draw(
            frame,
            ball_pos=ball_pos,
            detected_radius=detected_radius,
            trajectory_points=active_points,
            mask=debug_mask,
            is_predicted=is_predicted,
        )

        cv2.putText(vis, f"STATE: {state}  valid shots: {valid_shots}/{shot_count}",
                    (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, state_color, 2, cv2.LINE_AA)

        out.write(vis)  # always write full resolution

        if args.show:
            display = cv2.resize(vis, (1280, 720))  # resize only for display
            cv2.imshow("Basketball Tracker", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Done. Valid shots: {valid_shots} / Total detected: {shot_count}")
    print(f"[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()