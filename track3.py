# track.py (Clean version with enhanced trace continuity)
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
import datetime
import supervision as sv
from supervision.annotators.core import BoundingBoxAnnotator, LabelAnnotator, TraceAnnotator
from sort.sort import Sort
from scipy.signal import savgol_filter

__version__ = "Clean_Enhanced_Trace_V1"

# ---------------- CONFIG ----------------
class Config:
    VIDEO_PATH = "input/tclip2.mp4"
    MODEL_PATH = "trained-yolo/t2-biggerDS+augmentation/best.pt"
    OUTPUT_DIR = "outputs"
    TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, f"ball_tracking_output_{TIMESTAMP}.mp4")
    OUTPUT_CSV_PATH = os.path.join(OUTPUT_DIR, f"ball_coordinates_{TIMESTAMP}.csv")
    OUTPUT_PLOT_PATH = os.path.join(OUTPUT_DIR, f"plots_{TIMESTAMP}.png")

    CONFIDENCE_THRESHOLD = 0.05
    MIN_BOX_AREA = 15
    MAX_ASPECT_RATIO = 1.6
    
    COURT_WIDTH_METERS = 10.97

    # Enhanced trace settings for smoother visualization
    SORT_MAX_AGE = 90             # Increased for longer trace memory
    SORT_MIN_HITS = 2
    SORT_IOU_THRESHOLD = 0.15
    TRACE_LENGTH = 120            # Longer trace for better continuity

    UPSCALE_FACTOR = 2
    CLAHE_ENABLED = True
    USE_TTA = False


# ---------------- HELPERS ----------------
def has_boxes(detections):
    try:
        return (detections is not None) and hasattr(detections, "xyxy") and \
               (detections.xyxy is not None) and (detections.xyxy.shape[0] > 0)
    except Exception:
        return False


def apply_clahe(frame):
    """Enhance contrast with CLAHE (helps ball visibility)."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def interpolate_ball_positions(df):
    df_interp = df.copy()
    df_interp[['x_px', 'y_px']] = df_interp[['x_px', 'y_px']].interpolate(method='polynomial', order=2, limit_direction='both')
    df_interp = df_interp.bfill().ffill()
    return df_interp


def detect_ball_hits(df, window=5):
    df = df.copy()
    df['ball_hit'] = 0
    df['y_smooth'] = df['y_px'].rolling(window=window, min_periods=1, center=True).mean()
    df['y_grad'] = np.gradient(df['y_smooth'])
    
    for i in range(1, len(df) - 1):
        if df['y_grad'].iloc[i-1] < 0 and df['y_grad'].iloc[i] > 0:
            df.at[i, 'ball_hit'] = 1
    return df


# ---------------- ENHANCED TRACE ANNOTATOR ----------------
class EnhancedTraceAnnotator:
    def __init__(self, trace_length=120, thickness=2):
        self.trace_length = trace_length
        self.thickness = thickness
        self.positions = {}  # track_id -> list of (x, y) positions
        
    def annotate(self, scene, detections):
        if not has_boxes(detections):
            return scene
            
        for i in range(len(detections.xyxy)):
            track_id = detections.tracker_id[i]
            bbox = detections.xyxy[i]
            
            # Calculate center point
            center_x = int((bbox[0] + bbox[2]) / 2)
            center_y = int((bbox[1] + bbox[3]) / 2)
            
            # Update position history
            if track_id not in self.positions:
                self.positions[track_id] = []
            
            self.positions[track_id].append((center_x, center_y))
            
            # Keep only recent positions
            if len(self.positions[track_id]) > self.trace_length:
                self.positions[track_id] = self.positions[track_id][-self.trace_length:]
            
            # Draw trace
            points = self.positions[track_id]
            if len(points) > 1:
                # Draw trace with fading effect
                for j in range(1, len(points)):
                    # Calculate alpha based on recency
                    alpha = j / len(points)
                    color = (int(255 * alpha), int(100 * alpha), int(50 * alpha))  # Orange fade
                    
                    cv2.line(scene, points[j-1], points[j], color, 
                            max(1, int(self.thickness * alpha)))
        
        return scene


# ---------------- MAIN VIDEO PROCESS ----------------
def process_video(config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    model = YOLO(config.MODEL_PATH)
    cap = cv2.VideoCapture(config.VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps == 0:
        print("Warning: couldn't read FPS from video, defaulting to 30.0")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
    print("Model and video loaded successfully.")

    tracker = Sort(max_age=config.SORT_MAX_AGE, min_hits=config.SORT_MIN_HITS,
                   iou_threshold=config.SORT_IOU_THRESHOLD)
    box_annotator = BoundingBoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
    label_annotator = LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
    enhanced_trace_annotator = EnhancedTraceAnnotator(trace_length=config.TRACE_LENGTH)
    
    print(f"Enhanced tracker initialized (max_age={config.SORT_MAX_AGE}, "
          f"min_hits={config.SORT_MIN_HITS}, iou={config.SORT_IOU_THRESHOLD})")
    print(f"Enhanced trace length: {config.TRACE_LENGTH} frames")

    frame_idx = 0
    log_data = []

    print("\nStarting video processing...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Step 1: Detection
        if config.CLAHE_ENABLED:
            frame = apply_clahe(frame)
        upscale = config.UPSCALE_FACTOR
        frame_resized = cv2.resize(frame, None, fx=upscale, fy=upscale)
        results = model(frame_resized, conf=config.CONFIDENCE_THRESHOLD, verbose=False, augment=config.USE_TTA)[0]
        
        detections = sv.Detections.from_ultralytics(results)
        detections.xyxy /= upscale
        detections = filter_detections(detections, config)

        # Step 2: Feed detections to tracker
        if has_boxes(detections):
            detections_for_sort = np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))
        else:
            detections_for_sort = np.empty((0, 5))

        # Step 3: Get tracked objects
        tracked_objects = tracker.update(detections_for_sort)
        
        # Step 4: Select best object (same logic as original)
        tracked_detections = sv.Detections.empty()
        if tracked_objects is not None and tracked_objects.size > 0:
            last_known_position = None
            # Find the last valid logged position
            for i in range(len(log_data) - 1, -1, -1):
                if log_data[i][2] is not None:
                    last_known_position = np.array([log_data[i][2], log_data[i][3]])
                    break
            
            best_object = None
            if last_known_position is not None:
                tracked_centers = np.column_stack([
                    (tracked_objects[:, 0] + tracked_objects[:, 2]) / 2.0,
                    (tracked_objects[:, 1] + tracked_objects[:, 3]) / 2.0
                ])
                distances = np.linalg.norm(tracked_centers - last_known_position, axis=1)
                closest_idx = np.argmin(distances)
                best_object = tracked_objects[closest_idx:closest_idx + 1]
            else:
                areas = (tracked_objects[:, 2] - tracked_objects[:, 0]) * \
                        (tracked_objects[:, 3] - tracked_objects[:, 1])
                min_area_idx = np.argmin(areas)
                best_object = tracked_objects[min_area_idx:min_area_idx + 1]
            
            if best_object is not None:
                tracked_detections = sv.Detections(
                    xyxy=best_object[:, :4],
                    tracker_id=np.array([1])  # Consistent ID
                )

        # Step 5: Logging and annotation
        if has_boxes(tracked_detections):
            log_data.extend(get_log_data(tracked_detections, frame_idx))
        else:
            log_data.append([frame_idx, None, None, None])

        # Enhanced annotation with better trace
        annotated_frame = annotate_frame(frame, tracked_detections,
                                       box_annotator, label_annotator, enhanced_trace_annotator)
        
        out.write(annotated_frame)
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()
    out.release()
    print(f"Video processing finished. Output video saved to '{config.OUTPUT_VIDEO_PATH}'")
    return log_data, frame_idx, fps, w


# ---------------- FILTER + ANNOTATE ----------------
def filter_detections(detections, config):
    if not has_boxes(detections): 
        return detections
    mask = np.ones(detections.xyxy.shape[0], dtype=bool)
    for i, box in enumerate(detections.xyxy):
        area = (box[2] - box[0]) * (box[3] - box[1])
        if box[3] - box[1] > 0:
            aspect_ratio = (box[2] - box[0]) / (box[3] - box[1])
            if area < config.MIN_BOX_AREA or not (1 / config.MAX_ASPECT_RATIO < aspect_ratio < config.MAX_ASPECT_RATIO):
                mask[i] = False
        else:
            mask[i] = False
    return detections[mask]


def annotate_frame(frame, tracked_detections, box_annotator, label_annotator, enhanced_trace_annotator):
    """Clean annotation with enhanced trace continuity."""
    annotated = frame.copy()
    
    # Always update trace first (even if no current detections, maintains trail)
    annotated = enhanced_trace_annotator.annotate(annotated, tracked_detections)
    
    # Only draw boxes and labels for actual detections
    if has_boxes(tracked_detections):
        annotated = box_annotator.annotate(scene=annotated, detections=tracked_detections)
        labels = [f"Ball {tracker_id}" for tracker_id in tracked_detections.tracker_id]
        annotated = label_annotator.annotate(scene=annotated, detections=tracked_detections, labels=labels)
    
    return annotated


def get_log_data(tracked_detections, frame_idx):
    """Simple 4-column logging - no changes here."""
    if not has_boxes(tracked_detections): 
        return []
    
    xyxy = tracked_detections.xyxy[0]
    tracker_id = 1
    
    cx = int((xyxy[0] + xyxy[2]) / 2)
    cy = int((xyxy[1] + xyxy[3]) / 2)
    return [[frame_idx, tracker_id, cx, cy]]


# ---------------- SAVE + PLOT ----------------
def save_and_plot_results(log_data, frame_idx, config, fps, frame_width_px):
    if not log_data:
        print("No tracking data found to save or plot.")
        return

    # Simple 4-column DataFrame
    df = pd.DataFrame(log_data, columns=["frame", "track_id", "x_px", "y_px"])
    
    original_detections = df['x_px'].notna().sum()
    
    if original_detections == 0:
        print("No valid tracking data was logged.")
        return
        
    df = interpolate_ball_positions(df)

    if len(df) > 11:
        df['x_px_smooth'] = savgol_filter(df['x_px'], window_length=11, polyorder=3)
        df['y_px_smooth'] = savgol_filter(df['y_px'], window_length=11, polyorder=3)
    else: 
        df['x_px_smooth'] = df['x_px']
        df['y_px_smooth'] = df['y_px']
    
    df = detect_ball_hits(df)

    df.to_csv(config.OUTPUT_CSV_PATH, index=False)
    print(f"Coordinate data saved to '{config.OUTPUT_CSV_PATH}'")
    
    print(f"Ball actually detected in {original_detections} frames. Enhanced trace provides better continuity.")

    scale_m_per_px = config.COURT_WIDTH_METERS / frame_width_px
    df['x_m'] = df['x_px_smooth'] * scale_m_per_px
    df['y_m'] = df['y_px_smooth'] * scale_m_per_px

    group = df.sort_values(by='frame').copy()
    dx = group['x_m'].diff()
    dy = group['y_m'].diff()
    dt = group['frame'].diff() / fps
    
    speed_df = pd.DataFrame()
    if dt.sum() > 0:
        dt = dt.replace(0, np.nan)
        instant_speed_mps = np.sqrt(dx ** 2 + dy ** 2) / dt
        instant_speed_kmh = instant_speed_mps * 3.6
        speed_df = pd.DataFrame({'frame': group['frame'], 'speed_kmph': instant_speed_kmh}).dropna()
        speed_df['speed_kmph'] = speed_df['speed_kmph'].clip(0, 250)
        speed_df['speed_kmph_smoothed'] = speed_df['speed_kmph'].rolling(5, min_periods=1, center=True).median()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.invert_yaxis()
    ax1.set_xlabel("X (pixels)")
    ax1.set_ylabel("Y (pixels)")
    ax1.set_title("Ball Trajectory (Smoothed)")
    ax1.grid(True)
    ax1.plot(df["x_px_smooth"], df["y_px_smooth"], "o-", markersize=3, label="Ball Trajectory")
    hit_points = df[df['ball_hit'] == 1]
    ax1.scatter(hit_points["x_px"], hit_points["y_px"], c="red", marker="x", s=80, label="Hit")
    ax1.legend()

    if not speed_df.empty:
        avg_speed = speed_df['speed_kmph_smoothed'].mean()
        peak_speed = speed_df['speed_kmph_smoothed'].max()
        ax2.set_xlabel("Frame")
        ax2.set_ylabel("Speed (km/h)")
        ax2.set_title("Ball Speed Over Time")
        ax2.grid(True)
        ax2.plot(speed_df["frame"], speed_df["speed_kmph_smoothed"], "b.-", label="Instantaneous Speed")
        ax2.axhline(avg_speed, color="r", linestyle="--", label=f"Avg Speed: {avg_speed:.2f} km/h")
        ax2.legend()
        print(f"Average Speed: {avg_speed:.2f} km/h | Peak Speed: {peak_speed:.2f} km/h")
    else:
        print("Could not compute speeds.")
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_PLOT_PATH)
    print(f"Plot saved to '{config.OUTPUT_PLOT_PATH}'")
    plt.show()


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print(f"Starting Tennis Ball Tracker... (Version: {__version__})")
    config = Config()
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.VIDEO_PATH):
        print("Error: Model or video file not found. Please check the paths in the Config section.")
    else:
        try:
            import scipy
        except ImportError:
            print("Error: Scipy is not installed. Please run 'pip install scipy'")
            exit()
            
        try:
            log_data, total_frames, fps, width = process_video(config)
            save_and_plot_results(log_data, total_frames, config, fps, width)
        except Exception as e:
            import traceback
            print("Error during processing:", e)
            traceback.print_exc()