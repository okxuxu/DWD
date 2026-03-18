# Motion Scoring System for TouchDesigner + MediaPipe

## Overview
This project is a real-time body motion scoring system built in **TouchDesigner** using **MediaPipe Pose** landmark input.

Instead of checking whether the audience copies a predefined gesture, the system evaluates:

- **Stability**: how far the body deviates from a normalized neutral standing pose
- **Speed**: how fast body landmarks move over time
- **Overall score**: a weighted combination of stability and speed

The coordinate system is **body-centered**, not screen-centered. Each frame is normalized relative to the audience's own body center and body scale, so the score is less affected by where the person stands in the frame.

---

## Core Logic

### 1. Dynamic local coordinate system
For each frame, the system computes a body center using:

- left shoulder
- right shoulder
- left hip
- right hip

All landmark coordinates are converted into a local coordinate system centered on that body center.

### 2. Scale normalization
To reduce differences caused by body size and camera distance, the system normalizes each frame using a body scale derived from:

- shoulder width
- hip width
- torso length

### 3. Stability score
`stability` measures how much the current normalized pose differs from a neutral standing reference pose.

Important detail:
- In this system, **larger deviation = higher stability score**
- So the displayed “stability” is actually closer to **instability / movement amplitude / postural deviation**

Extra improvements included:
- weighted landmarks
- reduced Z-axis influence
- baseline subtraction for personal standing bias
- deadzone for tiny noise
- smoothing of center and scale

### 4. Speed score
`speed` measures how quickly landmarks move from one frame to the next.

Extra improvements included:
- small-motion thresholding
- outlier clamping
- activity ratio bonus
- short-window averaging
- EMA smoothing

### 5. Overall score
The final score is:

`overall = 0.2 * stability + 0.8 * speed`

This can be adjusted depending on whether you want to reward larger body deviation or faster movement more strongly.

---

## Main Files

### `score_logic`
This is the core scoring module.

It is responsible for:
- reading 33 MediaPipe landmarks from `dattochop1`
- converting them into a local normalized body coordinate system
- computing stability
- computing speed
- smoothing the real-time values
- returning the current frame score

Key exported functions:
- `score_current(now_time)`
- `reset_runtime()`

### `score_exec`
This is the frame callback script.

It is responsible for:
- calling `score_logic.score_current()` each frame
- updating the UI text fields
- pushing scores into `verify_logic2`
- controlling idle / running / finished state behavior
- deciding when the test is complete based on average overall score

### `verify_logic2`
This is the runtime state manager.

It typically handles:
- current state (`idle`, `running`, `finished`)
- score accumulation
- running averages
- test timing
- finalization rules

---

## TouchDesigner Node Expectations

This project assumes the following important operators exist:

- `/project1/pose_tracking/dattochop1`
- `/project1/pose_tracking/score_logic`
- `/project1/pose_tracking/score_exec`
- `/project1/pose_tracking/verify_logic2`
- `/project1/pose_tracking/score_out`
- `/project1/pose_tracking/ui_live_score_text`
- `/project1/pose_tracking/ui_mode_text`
- `/project1/pose_tracking/ui_instruction_text`
- `/project1/pose_tracking/ui_final_score_text`
- `/project1/pose_tracking/ui_final_time_text`
- `/project1/pose_tracking/ui_accuracy_text`
- `/project1/pose_tracking/ui_stability_text`
- `/project1/pose_tracking/ui_reaction_text`

---

## Input Requirements

`dattochop1` must provide these channels:

- `x`
- `y`
- `z`
- `visibility`

Each channel must contain **33 samples**, matching the MediaPipe Pose landmark order.

If the core torso points are not visible enough, the frame is treated as invalid.

---

## UI Meaning

During the running state, the current setup displays:

- **Live score** = current frame overall score
- **Accuracy text** = current frame stability score
- **Reaction text** = current frame speed score
- **Stability text** = running average overall score

Note: the word “stability” in the UI may not match the mathematical meaning exactly. Internally it represents deviation from a neutral standing pose.

### Right-click the mouse, display "view", and then click on the pop-up window to start the test.

![1773861993471](1773861993471.png)

