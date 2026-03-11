import json
import math
from pathlib import Path

FPS = 30

# 整体放慢倍率：越大越慢
SLOW_FACTOR = 1.8

BONES = [
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
    ["left_hip", "left_knee"],
    ["left_knee", "left_ankle"],
    ["right_hip", "right_knee"],
    ["right_knee", "right_ankle"]
]


def lerp(a, b, t):
    return a + (b - a) * t


def ease_in_out(t):
    return 0.5 - 0.5 * math.cos(math.pi * t)


def interp_pose(pose_a, pose_b, t):
    t = ease_in_out(t)
    out = {}
    for k in pose_a.keys():
        a = pose_a[k]
        b = pose_b[k]
        out[k] = [
            lerp(a[0], b[0], t),
            lerp(a[1], b[1], t),
            lerp(a[2], b[2], t),
        ]
    return out


def add_micro_motion(pose, t):
    sway = math.sin(t * 2.0 * math.pi * 0.18) * 0.01
    breathe = math.sin(t * 2.0 * math.pi * 0.30) * 0.008
    hand = math.sin(t * 2.0 * math.pi * 0.90) * 0.006
    foot = math.sin(t * 2.0 * math.pi * 0.70) * 0.004

    for k in pose.keys():
        pose[k][0] += sway
        pose[k][2] += breathe * 0.5

    pose["left_wrist"][1] += hand
    pose["right_wrist"][1] -= hand * 0.8
    pose["left_ankle"][2] += foot
    pose["right_ankle"][2] -= foot * 0.8
    return pose


def flip_y(pose):
    out = {}
    for k, v in pose.items():
        out[k] = [v[0], -v[1], v[2]]
    return out


# 基础站立
BASE = {
    "left_hip":      [-0.18,  0.00, 0.00],
    "right_hip":     [ 0.18,  0.00, 0.00],
    "left_knee":     [-0.18, -0.42, 0.02],
    "right_knee":    [ 0.18, -0.42, 0.02],
    "left_ankle":    [-0.18, -0.86, 0.03],
    "right_ankle":   [ 0.18, -0.86, 0.03],

    "left_shoulder": [-0.28,  0.62, 0.00],
    "right_shoulder":[ 0.28,  0.62, 0.00],
    "left_elbow":    [-0.46,  0.36, 0.02],
    "right_elbow":   [ 0.46,  0.36, 0.02],
    "left_wrist":    [-0.56,  0.12, 0.03],
    "right_wrist":   [ 0.56,  0.12, 0.03],
}

# 双手上举
ARMS_UP = {
    **BASE,
    "left_elbow":    [-0.34, 0.90, 0.05],
    "right_elbow":   [ 0.34, 0.90, 0.05],
    "left_wrist":    [-0.18, 1.20, 0.08],
    "right_wrist":   [ 0.18, 1.20, 0.08],
}

# 双手侧平举
ARMS_OPEN = {
    **BASE,
    "left_elbow":    [-0.68, 0.66, 0.02],
    "right_elbow":   [ 0.68, 0.66, 0.02],
    "left_wrist":    [-0.98, 0.70, 0.04],
    "right_wrist":   [ 0.98, 0.70, 0.04],
}

# 左侧弓步+右手斜上
LEFT_LUNGE_REACH = {
    "left_hip":      [-0.28, -0.02, 0.06],
    "right_hip":     [ 0.10,  0.02, -0.02],
    "left_knee":     [-0.40, -0.48, 0.14],
    "right_knee":    [ 0.16, -0.36, -0.02],
    "left_ankle":    [-0.50, -0.92, 0.18],
    "right_ankle":   [ 0.20, -0.82, -0.03],

    "left_shoulder": [-0.38, 0.60, 0.08],
    "right_shoulder":[ 0.12, 0.70, 0.02],
    "left_elbow":    [-0.54, 0.42, 0.10],
    "right_elbow":   [ 0.30, 0.96, 0.12],
    "left_wrist":    [-0.62, 0.18, 0.10],
    "right_wrist":   [ 0.48, 1.20, 0.16],
}

# 轻微下蹲+双臂前伸
SQUAT_FORWARD = {
    "left_hip":      [-0.20, -0.12, 0.08],
    "right_hip":     [ 0.20, -0.12, 0.08],
    "left_knee":     [-0.26, -0.42, 0.20],
    "right_knee":    [ 0.26, -0.42, 0.20],
    "left_ankle":    [-0.24, -0.82, 0.06],
    "right_ankle":   [ 0.24, -0.82, 0.06],

    "left_shoulder": [-0.26, 0.48, 0.10],
    "right_shoulder":[ 0.26, 0.48, 0.10],
    "left_elbow":    [-0.46, 0.40, 0.24],
    "right_elbow":   [ 0.46, 0.40, 0.24],
    "left_wrist":    [-0.62, 0.34, 0.34],
    "right_wrist":   [ 0.62, 0.34, 0.34],
}

# 右抬膝+左手上举
RIGHT_KNEE_UP = {
    "left_hip":      [-0.16, 0.00, 0.00],
    "right_hip":     [ 0.18, 0.08, 0.10],
    "left_knee":     [-0.16, -0.40, 0.02],
    "right_knee":    [ 0.36, -0.02, 0.28],
    "left_ankle":    [-0.16, -0.84, 0.04],
    "right_ankle":   [ 0.42,  0.18, 0.34],

    "left_shoulder": [-0.24, 0.66, 0.00],
    "right_shoulder":[ 0.28, 0.62, 0.02],
    "left_elbow":    [-0.10, 0.98, 0.08],
    "right_elbow":   [ 0.58, 0.58, 0.14],
    "left_wrist":    [ 0.02, 1.22, 0.12],
    "right_wrist":   [ 0.82, 0.62, 0.22],
}

# 星形展开
STAR_STEP = {
    "left_hip":      [-0.26, 0.02, 0.00],
    "right_hip":     [ 0.26, 0.02, 0.00],
    "left_knee":     [-0.34, -0.36, 0.02],
    "right_knee":    [ 0.34, -0.36, 0.02],
    "left_ankle":    [-0.46, -0.82, 0.04],
    "right_ankle":   [ 0.46, -0.82, 0.04],

    "left_shoulder": [-0.34, 0.64, 0.02],
    "right_shoulder":[ 0.34, 0.64, 0.02],
    "left_elbow":    [-0.70, 0.82, 0.06],
    "right_elbow":   [ 0.70, 0.82, 0.06],
    "left_wrist":    [-0.96, 0.96, 0.08],
    "right_wrist":   [ 0.96, 0.96, 0.08],
}

# 身体扭转
TWIST = {
    "left_hip":      [-0.14, -0.02, 0.08],
    "right_hip":     [ 0.22,  0.02, -0.06],
    "left_knee":     [-0.16, -0.40, 0.08],
    "right_knee":    [ 0.28, -0.38, -0.04],
    "left_ankle":    [-0.16, -0.84, 0.08],
    "right_ankle":   [ 0.34, -0.82, -0.04],

    "left_shoulder": [-0.18, 0.64, 0.14],
    "right_shoulder":[ 0.32, 0.60, -0.10],
    "left_elbow":    [-0.02, 0.74, 0.24],
    "right_elbow":   [ 0.18, 0.34, -0.18],
    "left_wrist":    [ 0.08, 0.56, 0.34],
    "right_wrist":   [-0.02, 0.10, -0.30],
}

SEQUENCE = [
    (BASE, 1.6),
    (ARMS_UP, 2.2),
    (ARMS_OPEN, 2.0),
    (LEFT_LUNGE_REACH, 2.4),
    (BASE, 1.6),
    (SQUAT_FORWARD, 2.4),
    (BASE, 1.6),
    (RIGHT_KNEE_UP, 2.4),
    (BASE, 1.6),
    (STAR_STEP, 2.2),
    (TWIST, 2.2),
    (ARMS_UP, 2.0),
    (BASE, 2.0),
]


def build_frames():
    frames = []
    current_time = 0.0

    for i in range(len(SEQUENCE) - 1):
        pose_a, dur = SEQUENCE[i]
        pose_b, _ = SEQUENCE[i + 1]

        segment_duration = dur * SLOW_FACTOR
        segment_frames = max(1, int(segment_duration * FPS))

        for f in range(segment_frames):
            t01 = f / segment_frames
            pose = interp_pose(pose_a, pose_b, t01)
            pose = add_micro_motion(pose, current_time)
            pose = flip_y(pose)

            frames.append({
                "t": round(current_time, 6),
                "joints": pose
            })
            current_time += 1.0 / FPS

    return frames


def main():
    frames = build_frames()
    payload = {
        "fps": FPS,
        "bones": BONES,
        "frames": frames
    }

    out_path = Path("reference_motion.json")
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {len(frames)} frames to {out_path.resolve()}")


if __name__ == "__main__":
    main()