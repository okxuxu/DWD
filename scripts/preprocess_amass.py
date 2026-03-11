import argparse
import json
from pathlib import Path

import numpy as np
import torch
import smplx

# SMPL / SMPL-H 常见 24 关节索引
JOINT_IDX = {
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21
}

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


def decode_gender(g):
    if isinstance(g, bytes):
        return g.decode("utf-8")
    if isinstance(g, np.ndarray):
        if g.ndim == 0:
            v = g.item()
            return decode_gender(v)
        if len(g) > 0:
            return decode_gender(g[0])
    return str(g)


def tensor(x):
    return torch.tensor(x, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="AMASS .npz path")
    parser.add_argument("--output", default="reference_motion.json", help="output json path")
    parser.add_argument("--model-dir", required=True, help="SMPL-H model directory")
    parser.add_argument("--fps", type=int, default=30, help="target fps after downsample")
    parser.add_argument("--gender", default=None, help="override gender: male/female/neutral")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)

    poses = data["poses"].astype(np.float32)
    trans = data["trans"].astype(np.float32)
    betas = data["betas"].astype(np.float32)[:10]
    mocap_fps = float(data["mocap_framerate"])
    gender = args.gender or decode_gender(data.get("gender", "neutral"))

    # AMASS/SMPL-H 常见布局:
    # 0:3 global_orient
    # 3:66 body_pose
    # 66:111 left_hand_pose
    # 111:156 right_hand_pose
    if poses.shape[1] < 156:
      raise ValueError(f"The current script is written based on the SMPL-H layout, but the dimension of poses is less than 156, which is actually {poses.shape[1]}")

    model = smplx.create(
        model_path=args.model_dir,
        model_type="smplh",
        gender=gender,
        use_pca=False,
        num_betas=10,
        batch_size=1
    )

    step = max(1, int(round(mocap_fps / args.fps)))
    frames = []

    for i in range(0, len(poses), step):
        p = poses[i]

        global_orient = tensor(p[:3]).unsqueeze(0)
        body_pose = tensor(p[3:66]).unsqueeze(0)
        left_hand_pose = tensor(p[66:111]).unsqueeze(0)
        right_hand_pose = tensor(p[111:156]).unsqueeze(0)
        transl = tensor(trans[i]).unsqueeze(0)
        betas_t = tensor(betas).unsqueeze(0)

        with torch.no_grad():
            out = model(
                betas=betas_t,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                transl=transl,
                return_verts=False
            )

        joints = out.joints[0].cpu().numpy()

        frame = {
            "t": len(frames) / args.fps,
            "joints": {
                name: joints[idx].tolist()
                for name, idx in JOINT_IDX.items()
            }
        }
        frames.append(frame)

    payload = {
        "fps": args.fps,
        "bones": BONES,
        "frames": frames
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(frames)} frames to {out_path.resolve()}")


if __name__ == "__main__":
    main()