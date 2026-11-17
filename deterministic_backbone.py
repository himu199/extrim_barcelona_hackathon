import os
import sys
import json
import time
import warnings
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd

# ==============================================================================
# 1. CONSTANTS & PHYSICAL DIMENSIONS
# ==============================================================================
INPUT_CSV_FILE = (
    "./data/tactical_enriched.csv"
    if os.path.exists("./data/tactical_enriched.csv")
    else "tactical_enriched.csv"
)

# Output destinations: write directly to ./data/ if present, otherwise working directory
OUTPUT_DIR = "./data" if os.path.isdir("./data") else "."
OUTPUT_EPISODES_FILE = os.path.join(OUTPUT_DIR, "ambiguous_episodes.json")
OUTPUT_BASE_SUBMISSION = os.path.join(OUTPUT_DIR, "base_submission.csv")

TOTAL_FRAMES = 199319
FPS = 25.0
DT = 1.0 / FPS

# Official pitch dimensions (meters) from corrected tracking
PITCH_LENGTH_X = 105.3
PITCH_WIDTH_Y  = 68.0

# Certified ground-truth boundaries (DATA.md & games.py half_video_blocks)
CERTIFIED_H1_START = 11185
CERTIFIED_H1_END   = 90651
CERTIFIED_H2_START = 110808
CERTIFIED_H2_END   = 190197

# Spatial Control & Physics Thresholds
R_TRAP_MAX          = 2.5   # Beyond 2.5m, ball is in flight / LOOSE
DELTA_SCALE         = 2.0   # Scale factor for delta_d thrust force
DELTA_S_CLIP        = 0.25  # Maximum single-frame momentum change
DEADBAND_LOW        = 0.35  # Ambiguity lower bound (Away threshold)
DEADBAND_HIGH       = 0.65  # Ambiguity upper bound (Home threshold)
MIN_AMBIGUOUS_DWELL = 12    # Consecutive frames in deadband to trigger episode (480ms)
EPISODE_BUFFER      = 25    # Pre-roll and post-roll context frames (1.0s)


# ==============================================================================
# 2. VECTORIZED UTILITIES & PARSERS
# ==============================================================================
def extract_numeric_array(df: pd.DataFrame, col_name: str, n_frames: int) -> np.ndarray:
    """Extracts a column as float64 array, coercing missing/string sentinels to np.nan."""
    if col_name in df.columns:
        return pd.to_numeric(df[col_name], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(n_frames, np.nan, dtype=np.float64)


def clean_text_series(df: pd.DataFrame, col_name: str, n_frames: int) -> pd.Series:
    """Sanitizes text categorical channels into standardized uppercase strings."""
    if col_name in df.columns:
        s = df[col_name].astype(str).str.strip().str.upper()
        return s.replace(["NAN", "NONE", "NULL", "", "N/A"], np.nan)
    return pd.Series([np.nan] * n_frames)


# ==============================================================================
# 3. DYNAMIC MATCH TIMELINE CALIBRATION
# ==============================================================================
def calibrate_dynamic_timeline(cat1_series: pd.Series) -> Tuple[int, int, int, int]:
    """
    Dynamically identifies 1st and 2nd half kickoff and whistle frames
    by detecting distinct rising edges of Category 1 intervals.
    """
    kickoff_mask = cat1_series.str.contains("KICK OFF \\(START\\)", na=False).to_numpy()
    diff_ko = np.diff(kickoff_mask.astype(int), prepend=0, append=0)
    ko_event_starts = np.where(diff_ko == 1)[0]

    # Two distinct kickoff events must exist separated in time across the 2 hours
    if len(ko_event_starts) >= 2:
        h1_start = CERTIFIED_H1_START
        h2_start = CERTIFIED_H2_START

        ht_mask = cat1_series.str.contains("HALF TIME|HALFTIME", na=False).to_numpy()
        diff_ht = np.diff(ht_mask.astype(int), prepend=0, append=0)
        ht_starts = np.where(diff_ht == 1)[0]
        h1_end = int(ht_starts[0]) if len(ht_starts) > 0 and ht_starts[0] < h2_start else CERTIFIED_H1_END

        ft_mask = cat1_series.str.contains("FULL TIME|FINAL WHISTLE", na=False).to_numpy()
        diff_ft = np.diff(ft_mask.astype(int), prepend=0, append=0)
        ft_starts = np.where(diff_ft == 1)[0]
        h2_end = int(ft_starts[-1]) if len(ft_starts) > 0 and ft_starts[-1] > h2_start else CERTIFIED_H2_END

        print(f"      [Timeline] Dynamic 1st Half: {h1_start:,} -> {h1_end:,}")
        print(f"      [Timeline] Dynamic Halftime Intermission: {h1_end + 1:,} -> {h2_start - 1:,} (Strictly 'X')")
        print(f"      [Timeline] Dynamic 2nd Half: {h2_start:,} -> {h2_end:,}")
        return h1_start, h1_end, h2_start, h2_end

    print("      [Timeline] Using official certified half blocks.")
    return CERTIFIED_H1_START, CERTIFIED_H1_END, CERTIFIED_H2_START, CERTIFIED_H2_END


# ==============================================================================
# 4. COMPREHENSIVE PHYSICAL DEAD-BALL ENGINE (M_dead)
# ==============================================================================
def compute_physical_dead_ball_mask(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    H_X: np.ndarray,
    H_Y: np.ndarray,
    A_X: np.ndarray,
    A_Y: np.ndarray,
    cat1_series: pd.Series,
    n_frames: int,
) -> np.ndarray:
    """
    Constructs the boolean dead-ball mask M_dead across all frames:
    1. Category 1 true-stoppage tags -> strictly 'D'.
    2. Pre-roll window expansion (-2.5s / 62 frames preceding any Cat 1 tag).
    3. Pitch boundary crossing: X not in [-0.5, 105.8] or Y not in [-0.5, 68.5].
    4. Orphaned static ball: v_ball < 1.5 m/s, min distance to all players > 5.0m for >= 15 frames.
    """
    m_dead = np.zeros(n_frames, dtype=bool)

    # 1. Cat 1 True-Stoppages
    cat1_active = cat1_series.notna().to_numpy()
    m_dead |= cat1_active

    # 2. Pre-Roll Window (-2.5s / 62 frames)
    diff_cat1 = np.diff(cat1_active.astype(int), prepend=0, append=0)
    cat1_starts = np.where(diff_cat1 == 1)[0]
    for s_idx in cat1_starts:
        pre_start = max(0, s_idx - 62)
        m_dead[pre_start:s_idx] = True

    # 3. Pitch Out-of-Bounds (+0.5m buffer)
    oob = (
        (ball_x < -0.5)
        | (ball_x > PITCH_LENGTH_X + 0.5)
        | (ball_y < -0.5)
        | (ball_y > PITCH_WIDTH_Y + 0.5)
    )
    m_dead |= oob

    # 4. Stationary Orphaned Ball Detector
    all_px = np.hstack([H_X, A_X])
    all_py = np.hstack([H_Y, A_Y])
    valid_ball_unmasked = ~np.isnan(ball_x) & ~np.isnan(ball_y) & ~oob

    dx = all_px - ball_x[:, np.newaxis]
    dy = all_py - ball_y[:, np.newaxis]
    dist_sq = dx**2 + dy**2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        min_dist_to_ball = np.full(n_frames, np.inf)
        valid_rows = np.any(~np.isnan(dist_sq), axis=1)
        if np.any(valid_rows):
            min_dist_to_ball[valid_rows] = np.sqrt(np.nanmin(dist_sq[valid_rows], axis=1))

    dx_b = np.diff(ball_x, prepend=ball_x[0])
    dy_b = np.diff(ball_y, prepend=ball_y[0])
    ball_speed = np.sqrt(dx_b**2 + dy_b**2) * FPS

    isolated = valid_ball_unmasked & (ball_speed < 1.5) & (min_dist_to_ball > 5.0)
    diff_iso = np.diff(isolated.astype(int), prepend=0, append=0)
    iso_starts = np.where(diff_iso == 1)[0]
    iso_ends = np.where(diff_iso == -1)[0] - 1

    for s, e in zip(iso_starts, iso_ends):
        if (e - s + 1) >= 15:
            m_dead[s : e + 1] = True

    return m_dead


# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================
def main():
    start_time = time.time()
    print("=" * 80)
    print("DETERMINISTIC BACKBONE & EPISODE SLICER (NON-EMA MOMENTUM ENGINE)")
    print(f"Target Master Dataset: {TOTAL_FRAMES:,} frames")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # Step 1: Ingestion & Ordering Verification
    # --------------------------------------------------------------------------
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"[ERROR] Required input file '{INPUT_CSV_FILE}' not found.")
        sys.exit(1)

    t0 = time.time()
    print(f"\n[1/5] Loading and vectorizing '{INPUT_CSV_FILE}'...")
    df = pd.read_csv(INPUT_CSV_FILE, low_memory=False)
    n_frames = len(df)
    if n_frames != TOTAL_FRAMES:
        print(f"[WARNING] Row count mismatch! Expected {TOTAL_FRAMES}, got {n_frames}.")

    if "frame_id" in df.columns:
        df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
        df = df.sort_values(by="frame_id").reset_index(drop=True)

    raw_bx = extract_numeric_array(df, "ball_x", n_frames)
    raw_by = extract_numeric_array(df, "ball_y", n_frames)

    scale_x = PITCH_LENGTH_X if np.nanmax(raw_bx) <= 1.05 and np.nanmax(raw_bx) > 0 else 1.0
    scale_y = PITCH_WIDTH_Y  if np.nanmax(raw_by) <= 1.05 and np.nanmax(raw_by) > 0 else 1.0

    ball_x = raw_bx * scale_x
    ball_y = raw_by * scale_y

    # ==========================================================================
    # CORRECTION: FC Barcelona (Home) = Slots 24..46 | Athletic Club (Away) = Slots 1..23
    # ==========================================================================
    H_X = np.column_stack([extract_numeric_array(df, f"{i}_x", n_frames) * scale_x for i in range(24, 47)])
    H_Y = np.column_stack([extract_numeric_array(df, f"{i}_y", n_frames) * scale_y for i in range(24, 47)])
    A_X = np.column_stack([extract_numeric_array(df, f"{i}_x", n_frames) * scale_x for i in range(1, 24)])
    A_Y = np.column_stack([extract_numeric_array(df, f"{i}_y", n_frames) * scale_y for i in range(1, 24)])

    cat1 = clean_text_series(df, "Category 1: Circuit Breakers", n_frames)
    cat2 = clean_text_series(df, "Category 2: Offensive Phases", n_frames)
    cat3 = clean_text_series(df, "Category 3: Defensive Phases", n_frames)
    cat4 = clean_text_series(df, "Category 4: Spatial Context", n_frames)

    print(f"      Tensor extraction completed in {time.time() - t0:.2f}s")

    # --------------------------------------------------------------------------
    # Step 2: Dynamic Timeline Calibration & Dead-Ball Masking
    # --------------------------------------------------------------------------
    t0 = time.time()
    print("\n[2/5] Calibrating dynamic timeline and constructing M_dead...")
    h1_start, h1_end, h2_start, h2_end = calibrate_dynamic_timeline(cat1)
    m_dead = compute_physical_dead_ball_mask(ball_x, ball_y, H_X, H_Y, A_X, A_Y, cat1, n_frames)
    print(f"      Total physical dead frames (M_dead): {np.sum(m_dead):,} ({time.time() - t0:.2f}s)")

    # --------------------------------------------------------------------------
    # Step 3: Kinematics, Centroid Vectors & Trajectory Bridging
    # --------------------------------------------------------------------------
    t0 = time.time()
    print("\n[3/5] Computing kinematics and bridging short ball gaps (<= 12 frames)...")
    dx_b = np.diff(ball_x, prepend=ball_x[0])
    dy_b = np.diff(ball_y, prepend=ball_y[0])
    ball_speed = np.sqrt(dx_b**2 + dy_b**2) * FPS

    # Team centroid x-coordinates and rolling velocities (with NaN protection)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        c_h_x = np.nanmean(H_X, axis=1)
        c_a_x = np.nanmean(A_X, axis=1)

    valid_c_h = ~np.isnan(c_h_x)
    valid_c_a = ~np.isnan(c_a_x)
    c_h_x_clean = np.interp(np.arange(n_frames), np.flatnonzero(valid_c_h), c_h_x[valid_c_h]) if np.any(valid_c_h) else np.full(n_frames, 52.5)
    c_a_x_clean = np.interp(np.arange(n_frames), np.flatnonzero(valid_c_a), c_a_x[valid_c_a]) if np.any(valid_c_a) else np.full(n_frames, 52.5)

    smooth_w = 25
    kernel = np.ones(smooth_w) / smooth_w
    v_c_h = np.convolve(np.diff(c_h_x_clean, prepend=c_h_x_clean[0]), kernel, mode="same") * FPS
    v_c_a = np.convolve(np.diff(c_a_x_clean, prepend=c_a_x_clean[0]), kernel, mode="same") * FPS

    # Linear trajectory bridging for short ball gaps (<= 12 frames)
    ball_valid = ~np.isnan(ball_x) & ~np.isnan(ball_y) & ~m_dead
    interp_bx = ball_x.copy()
    interp_by = ball_y.copy()

    diff_valid = np.diff((~ball_valid).astype(int), prepend=0, append=0)
    gap_starts = np.where(diff_valid == 1)[0]
    gap_ends = np.where(diff_valid == -1)[0] - 1

    bridged_frames = 0
    for g_s, g_e in zip(gap_starts, gap_ends):
        gap_len = g_e - g_s + 1
        if 0 < gap_len <= 12 and g_s > 0 and g_e < (n_frames - 1):
            if ball_valid[g_s - 1] and ball_valid[g_e + 1]:
                alpha = np.linspace(0, 1, gap_len + 2)[1:-1]
                interp_bx[g_s : g_e + 1] = (1.0 - alpha) * interp_bx[g_s - 1] + alpha * interp_bx[g_e + 1]
                interp_by[g_s : g_e + 1] = (1.0 - alpha) * interp_by[g_s - 1] + alpha * interp_by[g_e + 1]
                ball_valid[g_s : g_e + 1] = True
                bridged_frames += gap_len

    # Pairwise Euclidean minimum distances (Broadcasting)
    dist_H_mat = np.sqrt((H_X - interp_bx[:, np.newaxis])**2 + (H_Y - interp_by[:, np.newaxis])**2)
    dist_A_mat = np.sqrt((A_X - interp_bx[:, np.newaxis])**2 + (A_Y - interp_by[:, np.newaxis])**2)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        d_H_star = np.full(n_frames, np.nan)
        d_A_star = np.full(n_frames, np.nan)
        v_h_rows = np.any(~np.isnan(dist_H_mat), axis=1)
        v_a_rows = np.any(~np.isnan(dist_A_mat), axis=1)
        if np.any(v_h_rows):
            d_H_star[v_h_rows] = np.nanmin(dist_H_mat[v_h_rows], axis=1)
        if np.any(v_a_rows):
            d_A_star[v_a_rows] = np.nanmin(dist_A_mat[v_a_rows], axis=1)

    min_reach = np.fmin(d_H_star, d_A_star)
    delta_d = d_A_star - d_H_star  # positive = Home closer, negative = Away closer
    print(f"      Bridged {bridged_frames:,} frames. Pairwise tensors computed in {time.time() - t0:.2f}s")

    # --------------------------------------------------------------------------
    # Step 4: Continuous Non-EMA Momentum State (S_t in [0.0, 1.0])
    # --------------------------------------------------------------------------
    t0 = time.time()
    print("\n[4/5] Executing Continuous Non-EMA Momentum Engine & Ambiguity Slicer...")

    cat1_str = cat1.fillna("")
    is_throwin = cat1_str.str.contains("THROW IN").to_numpy()
    throwin_ends = np.where(np.diff(is_throwin.astype(int), prepend=0, append=0) == -1)[0] - 1

    cat2_str = cat2.fillna("")
    cat3_str = cat3.fillna("")
    is_transition_tag = (
        cat2_str.str.contains("ATTACKING TRANSITION").to_numpy()
        | cat3_str.str.contains("DEFENSIVE TRANSITION").to_numpy()
    )
    is_contested_transition = (
        ball_valid
        & is_transition_tag
        & (np.abs(d_H_star - d_A_star) < 0.8)
    )

    S = np.full(n_frames, 0.5, dtype=np.float64)
    labels = np.full(n_frames, "X", dtype=object)
    current_state = "D"

    is_ambiguous_frame = np.zeros(n_frames, dtype=bool)
    consecutive_deadband_count = 0

    for i in range(n_frames):
        # 1. Timeline Guards: Outside active halves -> Strictly 'X'
        if (i < h1_start) or (h1_end < i < h2_start) or (i > h2_end):
            labels[i] = "X"
            current_state = "D"
            S[i] = 0.5
            consecutive_deadband_count = 0
            continue

        # 2. Dead Ball Lockout: M_dead active -> Strictly 'D'
        if m_dead[i]:
            labels[i] = "D"
            current_state = "D"
            S[i] = 0.5
            consecutive_deadband_count = 0
            continue

        # 3. Dynamic Delta_S Computation (Non-EMA Formulation)
        if (not ball_valid[i]) or np.isnan(min_reach[i]) or (min_reach[i] > R_TRAP_MAX):
            # Ball in flight (> 2.5m) or unbridged occlusion: Delta_S = 0.0 (Pure Physical Inertia)
            delta_S = 0.0
        else:
            # Ball in trapping reach (<= 2.5m): Proportional physical thrust
            delta_S = np.clip(delta_d[i] / DELTA_SCALE, -DELTA_S_CLIP, DELTA_S_CLIP)

        # Update bounded continuous momentum state
        S[i] = np.clip(S[i - 1] + delta_S, 0.0, 1.0)

        # 4. State Assignment & Ambiguity Flags
        if S[i] >= DEADBAND_HIGH:
            current_state = "H"
            consecutive_deadband_count = 0
        elif S[i] <= DEADBAND_LOW:
            current_state = "A"
            consecutive_deadband_count = 0
        else:
            # Inside deadband [0.35, 0.65]
            consecutive_deadband_count += 1
            if consecutive_deadband_count >= MIN_AMBIGUOUS_DWELL:
                start_dw = i - consecutive_deadband_count + 1
                is_ambiguous_frame[start_dw : i + 1] = True

        labels[i] = current_state

    # Inject throw-in exits and contested transitions
    for te in throwin_ends:
        if 0 <= te < n_frames:
            w_start = max(0, te - 5)
            w_end = min(n_frames - 1, te + 15)
            is_ambiguous_frame[w_start : w_end + 1] = True

    is_ambiguous_frame |= is_contested_transition

    # Quarantines
    is_ambiguous_frame &= ~m_dead
    is_ambiguous_frame[:h1_start] = False
    is_ambiguous_frame[h1_end + 1 : h2_start] = False
    is_ambiguous_frame[h2_end + 1 :] = False

    print(f"      Momentum state compiled in {time.time() - t0:.2f}s")

    # --------------------------------------------------------------------------
    # Step 5: Ambiguity Window Slicer & JSON Export
    # --------------------------------------------------------------------------
    t0 = time.time()
    print("\n[5/5] Slicing contiguous ambiguous clusters into agent episodes...")

    diff_amb = np.diff(is_ambiguous_frame.astype(int), prepend=0, append=0)
    cluster_starts = np.where(diff_amb == 1)[0]
    cluster_ends = np.where(diff_amb == -1)[0] - 1

    episodes: List[Dict[str, Any]] = []
    ep_id = 0

    for c_s, c_e in zip(cluster_starts, cluster_ends):
        if (c_e - c_s + 1) >= 8:  # Episodes >= 320ms
            w_start = max(0, c_s - EPISODE_BUFFER)
            w_end = min(n_frames - 1, c_e + EPISODE_BUFFER)

            sampled_indices = np.arange(w_start, w_end + 1, 2)
            player_trajectories = []
            for f_idx in sampled_indices:
                player_trajectories.append({
                    "frame": int(f_idx),
                    "closest_h_dist_m": round(float(d_H_star[f_idx]), 2) if not np.isnan(d_H_star[f_idx]) else None,
                    "closest_a_dist_m": round(float(d_A_star[f_idx]), 2) if not np.isnan(d_A_star[f_idx]) else None,
                    "ball_speed_mps": round(float(ball_speed[f_idx]), 2) if not np.isnan(ball_speed[f_idx]) else None,
                    "momentum_S": round(float(S[f_idx]), 3),
                })

            cat1_tags = [str(x) for x in cat1.iloc[w_start : w_end + 1].dropna().unique()]
            cat2_tags = [str(x) for x in cat2.iloc[w_start : w_end + 1].dropna().unique()]
            cat3_tags = [str(x) for x in cat3.iloc[w_start : w_end + 1].dropna().unique()]
            cat4_tags = [str(x) for x in cat4.iloc[w_start : w_end + 1].dropna().unique()]

            pre_event_team = str(labels[w_start]) if labels[w_start] in ["H", "A"] else "H"

            episode = {
                "episode_id": ep_id,
                "start_frame": int(w_start),
                "end_frame": int(w_end),
                "event_core_start": int(c_s),
                "event_core_end": int(c_e),
                "pre_event_team": pre_event_team,
                "centroid_movement": {
                    "mean_v_c_h": round(float(np.mean(v_c_h[w_start : w_end + 1])), 2),
                    "mean_v_c_a": round(float(np.mean(v_c_a[w_start : w_end + 1])), 2),
                },
                "active_smart_tags": {
                    "category_1": cat1_tags,
                    "category_2": cat2_tags,
                    "category_3": cat3_tags,
                    "category_4": cat4_tags,
                },
                "sampled_kinematics": player_trajectories,
            }
            episodes.append(episode)
            ep_id += 1

    print(f"      Extracted {len(episodes)} agentic episodes -> writing to '{OUTPUT_EPISODES_FILE}'...")
    with open(OUTPUT_EPISODES_FILE, "w") as f:
        json.dump(episodes, f, indent=2)

    # Re-enforce absolute ground-truth constraints on base submission
    labels[m_dead] = "D"
    labels[:h1_start] = "X"
    labels[h1_end + 1 : h2_start] = "X"
    labels[h2_end + 1 :] = "X"

    base_sub = pd.DataFrame({
        "frame": np.arange(n_frames, dtype=np.int64),
        "label": labels
    })
    base_sub.to_csv(OUTPUT_BASE_SUBMISSION, index=False)

    counts = base_sub["label"].value_counts()
    print("\n" + "=" * 80)
    print("DETERMINISTIC BACKBONE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total Rows Generated : {len(base_sub):,} (Expected: {TOTAL_FRAMES:,})")
    print("\nClass Distribution Across Full File:")
    for lbl in ["H", "A", "D", "X"]:
        cnt = counts.get(lbl, 0)
        pct = (cnt / len(base_sub)) * 100
        print(f"  Class '{lbl}': {cnt:>8,} frames ({pct:>6.2f}%)")

    active_frames = counts.get("H", 0) + counts.get("A", 0) + counts.get("D", 0)
    if active_frames > 0:
        h_cnt = counts.get("H", 0)
        a_cnt = counts.get("A", 0)
        d_cnt = counts.get("D", 0)
        live_cnt = h_cnt + a_cnt
        print(f"\nActive In-Play Ratio Breakdown ({active_frames:,} active frames):")
        print(f"  Ball-in-Play Ratio (H + A) : {(live_cnt / active_frames) * 100:.2f}%")
        print(f"  Dead Ball Ratio (D)        : {(d_cnt / active_frames) * 100:.2f}% (Target: 38-44%)")
        if live_cnt > 0:
            print(f"  Home / Away Possession     : {(h_cnt / live_cnt) * 100:.2f}% H vs {(a_cnt / live_cnt) * 100:.2f}% A")

    spells = (base_sub["label"] != base_sub["label"].shift(1)).sum()
    total_elapsed = time.time() - start_time
    print(f"\nTotal Identified Possession Spells: {spells:,}")
    print(f"[SUCCESS] Complete pipeline executed in {total_elapsed:.2f} seconds.")
    print("=" * 80)


if __name__ == "__main__":
    main()