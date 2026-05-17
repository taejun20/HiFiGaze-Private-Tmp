import os
from itertools import zip_longest

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

show_RGB = True
show_RGBT = False

# CSVs for RGB and RGBT evaluations
RGB_CSV = "csvs/Eval_RGB_v2_lr3e-4_train2,6,7,8,9,10,12,14,16,17,18,19,20,21,22,24,25_val1,11,13,15,26_Epoch4_ValLoss1.4222.csv"
RGBT_CSV = ""

# Video / layout settings
OUTPUT_DIR = "debug_videos"
FPS = 30  # Doubled from 10 to show twice as many frames in the same time
FRAME_WIDTH = 1400
FRAME_HEIGHT = 950

# Smartphone physical size (cm)
# Width is 7.1cm, height is 14.4cm (portrait orientation)
PHONE_WIDTH_CM = 7.1
PHONE_HEIGHT_CM = 14.4

# Phone drawing size (pixels)
PHONE_WIDTH_PX = 360
PHONE_HEIGHT_PX = int(PHONE_WIDTH_PX * (PHONE_HEIGHT_CM / PHONE_WIDTH_CM))

# Position: single centered phone
PHONE_ORIGIN = ((FRAME_WIDTH - PHONE_WIDTH_PX) // 2, (FRAME_HEIGHT - PHONE_HEIGHT_PX) // 2)


def cm_to_px(x_cm: float, y_cm: float, origin):
    """Convert cm coordinates (top-left origin) to pixel coords inside a phone frame."""
    x_px = int(origin[0] + (x_cm / PHONE_WIDTH_CM) * PHONE_WIDTH_PX)
    y_px = int(origin[1] + (y_cm / PHONE_HEIGHT_CM) * PHONE_HEIGHT_PX)
    x_px = int(np.clip(x_px, origin[0], origin[0] + PHONE_WIDTH_PX))
    y_px = int(np.clip(y_px, origin[1], origin[1] + PHONE_HEIGHT_PX))
    return x_px, y_px


def draw_phone(frame, origin, row):
    """Draw phone outline plus GT (black), RGB pred (red), RGBT pred (blue) for one merged row."""
    x0, y0 = origin
    cv2.rectangle(frame, (x0, y0), (x0 + PHONE_WIDTH_PX, y0 + PHONE_HEIGHT_PX), (0, 0, 0), 2)

    if row is None:
        cv2.putText(frame, "no data", (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return

    # use gt from whichever side is present
    if not pd.isna(row.get("gt_x_rgb")):
        gt_x_cm = float(row["gt_x_rgb"])
        gt_y_cm = float(row["gt_y_rgb"])
    elif not pd.isna(row.get("gt_x_rgbt")):
        gt_x_cm = float(row["gt_x_rgbt"])
        gt_y_cm = float(row["gt_y_rgbt"])
    else:
        # No GT data available
        cv2.putText(frame, "no GT data", (x0 + 10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return
    gt_px = cm_to_px(gt_x_cm, gt_y_cm, origin)
    cv2.circle(frame, gt_px, 6, (0, 0, 0), -1)  # black GT

    if show_RGB and not pd.isna(row.get("pred_x_rgb")) and not pd.isna(row.get("pred_y_rgb")):
        pred_rgb_px = cm_to_px(float(row["pred_x_rgb"]), float(row["pred_y_rgb"]), origin)
        cv2.circle(frame, pred_rgb_px, 6, (0, 0, 255), -1)  # red RGB

    if show_RGBT and not pd.isna(row.get("pred_x_rgbt")) and not pd.isna(row.get("pred_y_rgbt")):
        pred_rgbt_px = cm_to_px(float(row["pred_x_rgbt"]), float(row["pred_y_rgbt"]), origin)
        cv2.circle(frame, pred_rgbt_px, 6, (255, 0, 0), -1)  # blue RGBT

    frame_id = str(row['frame_id'])
    meta = f"subj {int(row['subject'])} | frame {frame_id.zfill(5)}"
    cv2.putText(frame, meta, (x0, y0 + PHONE_HEIGHT_PX + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    
    # Build coordinate text
    coord_text = f"GT ({gt_x_cm:.2f}, {gt_y_cm:.2f}) cm"
    
    if show_RGB and not pd.isna(row.get("pred_x_rgb")) and not pd.isna(row.get("pred_y_rgb")):
        rgb_x = float(row["pred_x_rgb"])
        rgb_y = float(row["pred_y_rgb"])
        coord_text += f"   RGB ({rgb_x:.2f}, {rgb_y:.2f}) cm"
    
    if show_RGBT and not pd.isna(row.get("pred_x_rgbt")) and not pd.isna(row.get("pred_y_rgbt")):
        rgbt_x = float(row["pred_x_rgbt"])
        rgbt_y = float(row["pred_y_rgbt"])
        coord_text += f"   RGBT ({rgbt_x:.2f}, {rgbt_y:.2f}) cm"
    
    cv2.putText(
        frame,
        coord_text,
        (x0, y0 + PHONE_HEIGHT_PX + 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        1,
    )


def load_sorted(csv_path: str, prefix: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Convert frame_id to int for proper numeric sorting, then back to string for consistency
    df["frame_id"] = df["frame_id"].astype(int).astype(str)
    df = df.sort_values(by=["subject", "frame_id"]).reset_index(drop=True)
    df = df.rename(
        columns={
            "pred_x": f"pred_x_{prefix}",
            "pred_y": f"pred_y_{prefix}",
            "gt_x": f"gt_x_{prefix}",
            "gt_y": f"gt_y_{prefix}",
        }
    )
    return df


def create_video_for_subject(subject: int, rgb_df: pd.DataFrame = None, rgbt_df: pd.DataFrame = None):
    rgb_sub = pd.DataFrame() if rgb_df is None else rgb_df[rgb_df["subject"] == subject]
    rgbt_sub = pd.DataFrame() if rgbt_df is None else rgbt_df[rgbt_df["subject"] == subject]

    if (rgb_df is None or rgb_sub.empty) and (rgbt_df is None or rgbt_sub.empty):
        print(f"Subject {subject}: no data in either CSV, skipping.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"debug_phone_rgb_vs_rgbt_p{subject}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

    try:
        if show_RGB and show_RGBT and rgb_df is not None and rgbt_df is not None and not rgb_sub.empty and not rgbt_sub.empty:
            # Merge both RGB and RGBT data
            merged = pd.merge(
                rgb_sub,
                rgbt_sub,
                on=["subject", "frame_id"],
                how="outer",
                suffixes=("_rgb", "_rgbt"),
            )
        elif show_RGB and rgb_df is not None and not rgb_sub.empty:
            # Only use RGB data
            merged = rgb_sub.copy()
        elif show_RGBT and rgbt_df is not None and not rgbt_sub.empty:
            # Only use RGBT data
            merged = rgbt_sub.copy()
        else:
            print(f"Subject {subject}: no data available for selected models, skipping.")
            return
        
        # Sort by frame_id (as integer for proper numeric sorting)
        merged["frame_id_int"] = merged["frame_id"].astype(int)
        merged = merged.sort_values(by="frame_id_int").reset_index(drop=True)
        merged = merged.drop(columns=["frame_id_int"])

        for _, merged_row in tqdm(merged.iterrows(), total=len(merged), desc=f"Rendering subject {subject}"):
            frame = np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 255, dtype=np.uint8)

            cv2.putText(
                frame,
                "(0cm,0cm) at top-left of phone frame",
                (FRAME_WIDTH // 2 - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            draw_phone(frame, PHONE_ORIGIN, merged_row)

            writer.write(frame)
    finally:
        writer.release()
        cv2.destroyAllWindows()

    print(f"Saved video to {output_path}")


if __name__ == "__main__":
    rgb_df_all = None
    rgbt_df_all = None
    subjects = set()
    
    # Filter settings
    TARGET_SUBJECT = 11
    MAX_FRAME_ID = 1198
    
    if show_RGB:
        try:
            rgb_df_all = load_sorted(RGB_CSV, "rgb")
            # Filter to subject 11 and frame_id <= 1198
            rgb_df_all = rgb_df_all[(rgb_df_all["subject"] == TARGET_SUBJECT) & (rgb_df_all["frame_id"].astype(int) <= MAX_FRAME_ID)]
            if not rgb_df_all.empty:
                subjects.add(TARGET_SUBJECT)
        except Exception as e:
            print(f"Warning: Could not load RGB CSV ({RGB_CSV}): {e}")
            print("Continuing without RGB data...")
    
    if show_RGBT:
        try:
            rgbt_df_all = load_sorted(RGBT_CSV, "rgbt")
            # Filter to subject 11 and frame_id <= 1198
            rgbt_df_all = rgbt_df_all[(rgbt_df_all["subject"] == TARGET_SUBJECT) & (rgbt_df_all["frame_id"].astype(int) <= MAX_FRAME_ID)]
            if not rgbt_df_all.empty:
                subjects.add(TARGET_SUBJECT)
        except Exception as e:
            print(f"Warning: Could not load RGBT CSV ({RGBT_CSV}): {e}")
            print("Continuing without RGBT data...")
    
    if not subjects:
        print("Error: No data loaded from either CSV. Please check your CSV file paths and show_RGB/show_RGBT flags.")
        exit(1)
    
    subjects = sorted(subjects)

    for subj in subjects:
        create_video_for_subject(subj, rgb_df_all, rgbt_df_all)

