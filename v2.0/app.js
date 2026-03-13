import {
  FilesetResolver,
  PoseLandmarker,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");

const guideCanvas = document.getElementById("guideCanvas");
const guideCtx = guideCanvas.getContext("2d");

const statusEl = document.getElementById("status");
const scoreValueEl = document.getElementById("scoreValue");
const taskNameEl = document.getElementById("taskName");
const modeValueEl = document.getElementById("modeValue");
const guideCaptionEl = document.getElementById("guideCaption");
const instructionBannerEl = document.getElementById("instructionBanner");

const startBtn = document.getElementById("startBtn");
const resetBtn = document.getElementById("resetBtn");
const modalResetBtn = document.getElementById("modalResetBtn");

const resultModal = document.getElementById("resultModal");
const finalScoreEl = document.getElementById("finalScore");
const finalVerdictEl = document.getElementById("finalVerdict");
const resultAccuracyEl = document.getElementById("resultAccuracy");
const resultStabilityEl = document.getElementById("resultStability");
const resultReactionEl = document.getElementById("resultReaction");
const finalTimeBigEl = document.getElementById("finalTimeBig");
const finalTimeTextEl = document.getElementById("finalTimeText");

let poseLandmarker = null;
let drawingUtils = null;

let lastVideoTime = -1;
let running = false;

const CALIBRATION_DURATION_MS = 1500;
const DEMO_DURATION_MS = 2600;
const TASK_DURATION_MS = 11000;
const HOLD_DURATION_MS = 900;

let state = "idle";
// idle | calibrating | demo | performing | finished

let globalStartTime = 0;
let stateStartTime = 0;

let calibrationFrames = [];
let baselinePose = null;

let currentTaskIndex = 0;
let holdStartTime = null;
let smoothedScore = 0;
let guideTime = 0;

let taskResults = [];

const TASKS = [
  {
    id: "airplane_arms",
    name: "Airplane Arms",
    instruction: "Stretch both arms sideways like airplane wings and hold them level.",
    guideCaption: "Both arms go straight out to the sides. Keep them wide and level.",
    demoType: "airplane_arms"
  },
  {
    id: "side_reach",
    name: "Side Reach",
    instruction: "Reach to the left, then reach to the right with your arm and upper body.",
    guideCaption:
      "First reach left, then reach right. Let the arm and torso travel together.",
    demoType: "side_reach"
  }
];

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: 1280,
      height: 720,
      facingMode: "user"
    },
    audio: false
  });

  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });

  await video.play();
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
  canvas.width = video.videoWidth || 1280;
  canvas.height = video.videoHeight || 720;
}

async function createPoseLandmarker() {
  statusEl.textContent = "Loading model...";

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "./model/pose_landmarker_full.task"
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });

  drawingUtils = new DrawingUtils(ctx);
  statusEl.textContent = "Ready";
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function easeInOut(t) {
  return 0.5 - 0.5 * Math.cos(Math.PI * clamp01(t));
}

function average(values) {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function distance2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function midpoint(a, b) {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2
  };
}

function averageLandmark(frames, idx) {
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let count = 0;

  for (const frame of frames) {
    const p = frame.landmarks[idx];
    if (!p) continue;
    if ((p.visibility ?? 1) < 0.45) continue;

    sx += p.x;
    sy += p.y;
    sz += p.z ?? 0;
    count++;
  }

  if (count === 0) return null;

  return {
    x: sx / count,
    y: sy / count,
    z: sz / count
  };
}

function computeBaselinePose(frames) {
  return {
    nose: averageLandmark(frames, 0),
    leftShoulder: averageLandmark(frames, 11),
    rightShoulder: averageLandmark(frames, 12),
    leftElbow: averageLandmark(frames, 13),
    rightElbow: averageLandmark(frames, 14),
    leftWrist: averageLandmark(frames, 15),
    rightWrist: averageLandmark(frames, 16),
    leftHip: averageLandmark(frames, 23),
    rightHip: averageLandmark(frames, 24),
    leftKnee: averageLandmark(frames, 25),
    rightKnee: averageLandmark(frames, 26),
    leftAnkle: averageLandmark(frames, 27),
    rightAnkle: averageLandmark(frames, 28)
  };
}

function getPoseScale(landmarks) {
  const ls = landmarks[11];
  const rs = landmarks[12];
  const lh = landmarks[23];
  const rh = landmarks[24];

  if (!ls || !rs || !lh || !rh) return 0.1;

  const shoulderWidth = distance2D(ls, rs);
  const torsoLength = distance2D(midpoint(ls, rs), midpoint(lh, rh));
  return Math.max(0.08, shoulderWidth * 0.75 + torsoLength * 1.0);
}

function setModeText() {
  const map = {
    idle: "Idle",
    calibrating: "Calibrating",
    demo: "Demo",
    performing: "Performing",
    finished: "Finished"
  };
  modeValueEl.textContent = map[state] ?? state;
}

function setCurrentTaskUI() {
  const task = TASKS[currentTaskIndex];

  if (!task) {
    taskNameEl.textContent = "Completed";
    guideCaptionEl.textContent = "Verification sequence finished.";
    return;
  }

  taskNameEl.textContent = task.name;
  guideCaptionEl.textContent = task.guideCaption;
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function clearGuideCanvas() {
  guideCtx.clearRect(0, 0, guideCanvas.width, guideCanvas.height);
}

function hideModal() {
  resultModal.classList.add("hidden");
}

function showModal() {
  resultModal.classList.remove("hidden");
}

function resetGuide() {
  guideTime = 0;
  drawGuide(0);
}

function startTest() {
  hideModal();
  running = true;
  globalStartTime = performance.now();
  stateStartTime = globalStartTime;
  state = "calibrating";

  calibrationFrames = [];
  baselinePose = null;
  currentTaskIndex = 0;
  holdStartTime = null;
  smoothedScore = 0;
  taskResults = [];
  lastVideoTime = -1;

  scoreValueEl.textContent = "0.0";
  setModeText();
  setCurrentTaskUI();
  instructionBannerEl.textContent = "Stand still briefly for calibration.";
  resetGuide();
}

function resetTest() {
  running = false;
  lastVideoTime = -1;
  state = "idle";
  calibrationFrames = [];
  baselinePose = null;
  currentTaskIndex = 0;
  holdStartTime = null;
  smoothedScore = 0;
  taskResults = [];

  setModeText();
  hideModal();

  scoreValueEl.textContent = "--";
  taskNameEl.textContent = "Waiting";
  guideCaptionEl.textContent = "A guide animation will appear here.";
  instructionBannerEl.textContent =
    "Stand where your full body is visible, then press Start Test.";

  clearCanvas();
  clearGuideCanvas();
  drawGuide(0);
}

function getVerdict(score) {
  if (score >= 80) return "HUMAN CONFIRMED";
  return "LIKELY HUMAN";
}

function finishTest() {
  running = false;
  state = "finished";
  setModeText();

  const accuracy = average(taskResults.map((r) => r.bestPoseScore * 100));
  const stability = average(taskResults.map((r) => r.stability * 100));
  const reaction = average(taskResults.map((r) => r.reactionScore * 100));
  const overall = average(taskResults.map((r) => r.finalTaskScore));

  finalScoreEl.textContent = overall.toFixed(1);
  finalVerdictEl.textContent = getVerdict(overall);
  resultAccuracyEl.textContent = accuracy.toFixed(1);
  resultStabilityEl.textContent = stability.toFixed(1);
  resultReactionEl.textContent = reaction.toFixed(1);

  const totalSec = (performance.now() - globalStartTime) / 1000;
  finalTimeBigEl.textContent = `${totalSec.toFixed(1)} sec`;
  finalTimeTextEl.textContent = `You spent ${totalSec.toFixed(1)} seconds proving that you are human.`;

  instructionBannerEl.textContent = "Verification complete.";
  taskNameEl.textContent = "Completed";

  showModal();
}

function moveToNextTask() {
  currentTaskIndex += 1;
  holdStartTime = null;
  smoothedScore = 0;

  if (currentTaskIndex >= TASKS.length) {
    finishTest();
    return;
  }

  state = "demo";
  stateStartTime = performance.now();
  setModeText();
  setCurrentTaskUI();
  resetGuide();
}

function startDemoPhase() {
  state = "demo";
  stateStartTime = performance.now();
  setModeText();
  setCurrentTaskUI();
  resetGuide();
}

function startPerformPhase() {
  state = "performing";
  stateStartTime = performance.now();
  holdStartTime = null;
  smoothedScore = 0;
  setModeText();

  const task = TASKS[currentTaskIndex];
  instructionBannerEl.textContent = task.instruction;
}

function drawResults(result) {
  clearCanvas();

  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  if (result.landmarks && result.landmarks.length > 0) {
    const landmarks = result.landmarks[0];

    drawingUtils.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
      lineWidth: 3,
      color: "rgba(0, 128, 255, 0.9)"
    });

    drawingUtils.drawLandmarks(landmarks, {
      radius: 4,
      color: "rgba(255, 0, 128, 0.95)",
      fillColor: "rgba(255, 0, 128, 0.95)"
    });
  }

  ctx.restore();
}

function getTaskRecord() {
  if (!taskResults[currentTaskIndex]) {
    taskResults[currentTaskIndex] = {
      taskId: TASKS[currentTaskIndex].id,
      startedAt: stateStartTime,
      firstMotionAt: null,
      firstReadyAt: null,
      bestPoseScore: 0,
      bestRawScore: 0,
      holdFrames: [],
      reactionScore: 0,
      stability: 0,
      finalTaskScore: 0,
      directionSequence: [],
      reachFrames: [],
      leftReachPeak: 0,
      rightReachPeak: 0
    };
  }
  return taskResults[currentTaskIndex];
}

function scoreAirplaneArms(landmarks, baseline) {
  const ls = landmarks[11];
  const rs = landmarks[12];
  const le = landmarks[13];
  const re = landmarks[14];
  const lw = landmarks[15];
  const rw = landmarks[16];

  if (
    !ls ||
    !rs ||
    !le ||
    !re ||
    !lw ||
    !rw ||
    !baseline?.leftWrist ||
    !baseline?.rightWrist
  ) {
    return { score: 0, poseScore: 0, holdReady: false, motionGate: 0 };
  }

  const scale = getPoseScale(landmarks);

  const leftOut = Math.abs(lw.x - ls.x);
  const rightOut = Math.abs(rw.x - rs.x);
  const spreadGate = clamp01((Math.min(leftOut, rightOut) - 0.1) / 0.16);

  const leftHorizontal = 1 - clamp01(Math.abs(lw.y - ls.y) / (0.22 * scale));
  const rightHorizontal = 1 - clamp01(Math.abs(rw.y - rs.y) / (0.22 * scale));
  const leftElbowHorizontal = 1 - clamp01(Math.abs(le.y - ls.y) / (0.24 * scale));
  const rightElbowHorizontal = 1 - clamp01(Math.abs(re.y - rs.y) / (0.24 * scale));
  const armSpread = clamp01((Math.abs(lw.x - rw.x) - 0.34) / 0.24);

  const leftMove = distance2D(lw, baseline.leftWrist) / Math.max(scale, 0.08);
  const rightMove = distance2D(rw, baseline.rightWrist) / Math.max(scale, 0.08);
  const motionGate = clamp01((Math.max(leftMove, rightMove) - 0.08) / 0.14);

  const poseScore =
    leftHorizontal * 0.22 +
    rightHorizontal * 0.22 +
    leftElbowHorizontal * 0.14 +
    rightElbowHorizontal * 0.14 +
    armSpread * 0.28;

  const score = 100 * spreadGate * poseScore;
  const holdReady = spreadGate > 0.68 && poseScore > 0.72;

  return { score, poseScore, holdReady, motionGate };
}

function updateDirectionSequence(record, dir) {
  if (dir === "center") return;
  const seq = record.directionSequence;
  const last = seq[seq.length - 1];

  if (dir !== last) {
    seq.push(dir);
    if (seq.length > 8) seq.shift();
  }
}

function sequenceMatches(record) {
  const seq = record.directionSequence.join(",");
  return seq.includes("left,right");
}

function scoreSideReach(landmarks, baseline, record) {
  const ls = landmarks[11];
  const rs = landmarks[12];
  const lw = landmarks[15];
  const rw = landmarks[16];
  const le = landmarks[13];
  const re = landmarks[14];
  const lh = landmarks[23];
  const rh = landmarks[24];

  if (
    !ls ||
    !rs ||
    !lw ||
    !rw ||
    !le ||
    !re ||
    !lh ||
    !rh ||
    !baseline?.leftWrist ||
    !baseline?.rightWrist ||
    !baseline?.leftShoulder ||
    !baseline?.rightShoulder ||
    !baseline?.leftHip ||
    !baseline?.rightHip ||
    !baseline?.leftElbow ||
    !baseline?.rightElbow
  ) {
    return { score: 0, poseScore: 0, holdReady: false, motionGate: 0 };
  }

  const scale = getPoseScale(landmarks);

  const shoulderMid = midpoint(ls, rs);
  const baseShoulderMid = midpoint(baseline.leftShoulder, baseline.rightShoulder);

  const hipMid = midpoint(lh, rh);
  const baseHipMid = midpoint(baseline.leftHip, baseline.rightHip);

  const torsoShift = (shoulderMid.x - baseShoulderMid.x) / Math.max(scale, 0.08);
  const hipShift = (hipMid.x - baseHipMid.x) / Math.max(scale, 0.08);

  const leftReachArm =
    ((baseline.leftWrist.x - lw.x) + Math.max(0, baseline.leftShoulder.x - ls.x)) /
    Math.max(scale, 0.08);
  const rightReachArm =
    ((rw.x - baseline.rightWrist.x) + Math.max(0, rs.x - baseline.rightShoulder.x)) /
    Math.max(scale, 0.08);

  const leftTorso = clamp01((-(torsoShift + hipShift * 0.8) - 0.01) / 0.12);
  const rightTorso = clamp01(((torsoShift + hipShift * 0.8) - 0.01) / 0.12);

  const leftArmOpen = clamp01((leftReachArm - 0.05) / 0.20);
  const rightArmOpen = clamp01((rightReachArm - 0.05) / 0.20);

  const leftElbowSupport = clamp01((baseline.leftElbow.x - le.x + 0.02) / 0.16);
  const rightElbowSupport = clamp01((re.x - baseline.rightElbow.x + 0.02) / 0.16);

  const leftReachScore =
    leftArmOpen * 0.6 + leftTorso * 0.25 + leftElbowSupport * 0.15;
  const rightReachScore =
    rightArmOpen * 0.6 + rightTorso * 0.25 + rightElbowSupport * 0.15;

  record.leftReachPeak = Math.max(record.leftReachPeak, leftReachScore);
  record.rightReachPeak = Math.max(record.rightReachPeak, rightReachScore);

  let dir = "center";
  if (leftReachScore > rightReachScore && leftReachScore > 0.48) dir = "left";
  if (rightReachScore > leftReachScore && rightReachScore > 0.48) dir = "right";
  updateDirectionSequence(record, dir);

  const patternScore = sequenceMatches(record)
    ? 1
    : clamp01((record.leftReachPeak + record.rightReachPeak) / 1.4);

  const amplitudeScore = clamp01(
    (Math.max(record.leftReachPeak, record.rightReachPeak) - 0.45) / 0.35
  );

  const bilateralScore = Math.min(
    1,
    Math.min(record.leftReachPeak, record.rightReachPeak) / 0.7
  );

  const poseScore =
    patternScore * 0.38 +
    bilateralScore * 0.26 +
    amplitudeScore * 0.20 +
    Math.max(leftTorso, rightTorso) * 0.16;

  const motionGate = clamp01(
    (Math.max(leftReachScore, rightReachScore, Math.abs(torsoShift)) - 0.18) / 0.35
  );

  const score = 100 * motionGate * poseScore;
  const holdReady = sequenceMatches(record) && bilateralScore > 0.62 && poseScore > 0.68;

  record.reachFrames.push(Math.max(leftReachScore, rightReachScore) * 100);
  if (record.reachFrames.length > 140) record.reachFrames.shift();

  return { score, poseScore, holdReady, motionGate };
}

function scoreCurrentTask(landmarks, baseline, record) {
  const task = TASKS[currentTaskIndex];
  if (!task) return { score: 0, poseScore: 0, holdReady: false, motionGate: 0 };

  if (task.id === "airplane_arms") {
    return scoreAirplaneArms(landmarks, baseline);
  }

  if (task.id === "side_reach") {
    return scoreSideReach(landmarks, baseline, record);
  }

  return { score: 0, poseScore: 0, holdReady: false, motionGate: 0 };
}

function finalizeCurrentTask() {
  const record = getTaskRecord();
  const heldAvg = record.holdFrames.length ? average(record.holdFrames) : record.bestRawScore;

  let reactionScore = 0.18;

  if (record.firstMotionAt != null) {
    const reactionMs = Math.max(0, record.firstMotionAt - record.startedAt);
    reactionScore = 1 - clamp01((reactionMs - 450) / 3200);
  } else if (record.bestPoseScore > 0) {
    reactionScore = clamp01(0.25 + record.bestPoseScore * 0.4);
  }

  const stability = clamp01((heldAvg || 0) / 100);

  record.reactionScore = reactionScore;
  record.stability = stability;
  record.finalTaskScore =
    record.bestPoseScore * 100 * 0.5 +
    stability * 100 * 0.3 +
    reactionScore * 100 * 0.2;

  moveToNextTask();
}

// ---------- Guide silhouette drawing ----------

function makeSilhouettePose(base) {
  const neck = base.neck;
  const head = base.head;
  const ls = base.leftShoulder;
  const rs = base.rightShoulder;
  const le = base.leftElbow;
  const re = base.rightElbow;
  const lw = base.leftWrist;
  const rw = base.rightWrist;
  const lh = base.leftHip;
  const rh = base.rightHip;
  const lk = base.leftKnee;
  const rk = base.rightKnee;
  const la = base.leftAnkle;
  const ra = base.rightAnkle;

  return {
    head,
    neck,
    leftArm: [ls, le, lw],
    rightArm: [rs, re, rw],
    torsoLeft: [ls, lh],
    torsoRight: [rs, rh],
    hipLine: [lh, rh],
    leftLeg: [lh, lk, la],
    rightLeg: [rh, rk, ra]
  };
}

function drawThickLine(a, b, width) {
  guideCtx.beginPath();
  guideCtx.moveTo(a.x, a.y);
  guideCtx.lineTo(b.x, b.y);
  guideCtx.lineWidth = width;
  guideCtx.stroke();
}

function drawPolyline(points, width) {
  if (!points || points.length < 2) return;
  guideCtx.beginPath();
  guideCtx.moveTo(points[0].x, points[0].y);
  for (let i = 1; i < points.length; i++) {
    guideCtx.lineTo(points[i].x, points[i].y);
  }
  guideCtx.lineWidth = width;
  guideCtx.stroke();
}

function drawSilhouette(base) {
  const p = makeSilhouettePose(base);

  guideCtx.save();
  guideCtx.lineCap = "round";
  guideCtx.lineJoin = "round";
  guideCtx.strokeStyle = "#111111";
  guideCtx.fillStyle = "#111111";

  // head
  guideCtx.beginPath();
  guideCtx.arc(p.head.x, p.head.y, 12, 0, Math.PI * 2);
  guideCtx.fill();

  // neck to shoulders
  drawThickLine(
    { x: p.leftArm[0].x, y: p.leftArm[0].y },
    { x: p.rightArm[0].x, y: p.rightArm[0].y },
    12
  );

  // upper torso
  drawThickLine(p.neck, midpoint(p.hipLine[0], p.hipLine[1]), 12);

  // torso sides
  drawPolyline(p.torsoLeft, 12);
  drawPolyline(p.torsoRight, 12);

  // arms
  drawPolyline(p.leftArm, 11);
  drawPolyline(p.rightArm, 11);

  // legs
  drawPolyline(p.leftLeg, 12);
  drawPolyline(p.rightLeg, 12);

  guideCtx.restore();
}

function createGuidePose(type, e, timeMs = 0) {
  const base = {
    head: { x: 0.5, y: 0.13 },
    neck: { x: 0.5, y: 0.23 },
    leftShoulder: { x: 0.42, y: 0.29 },
    rightShoulder: { x: 0.58, y: 0.29 },
    leftElbow: { x: 0.37, y: 0.43 },
    rightElbow: { x: 0.63, y: 0.43 },
    leftWrist: { x: 0.34, y: 0.57 },
    rightWrist: { x: 0.66, y: 0.57 },
    leftHip: { x: 0.45, y: 0.52 },
    rightHip: { x: 0.55, y: 0.52 },
    leftKnee: { x: 0.46, y: 0.70 },
    rightKnee: { x: 0.54, y: 0.70 },
    leftAnkle: { x: 0.47, y: 0.90 },
    rightAnkle: { x: 0.53, y: 0.90 }
  };

  if (type === "airplane_arms") {
    base.leftElbow.x = lerp(0.37, 0.23, e);
    base.leftWrist.x = lerp(0.34, 0.08, e);
    base.rightElbow.x = lerp(0.63, 0.77, e);
    base.rightWrist.x = lerp(0.66, 0.92, e);

    base.leftElbow.y = lerp(0.43, 0.30, e);
    base.rightElbow.y = lerp(0.43, 0.30, e);
    base.leftWrist.y = lerp(0.57, 0.31, e);
    base.rightWrist.y = lerp(0.57, 0.31, e);
  }

  if (type === "side_reach") {
    const cycle = (timeMs % 3200) / 3200;
    let shift = 0;
    let reachLeft = true;

    if (cycle < 0.5) {
      const t = easeInOut(cycle / 0.5);
      shift = lerp(0, -0.08, t);
      reachLeft = true;
    } else {
      const t = easeInOut((cycle - 0.5) / 0.5);
      shift = lerp(-0.08, 0.08, t);
      reachLeft = false;
    }

    base.head.x += shift * 0.35;
    base.neck.x += shift * 0.45;
    base.leftShoulder.x += shift * 0.65;
    base.rightShoulder.x += shift * 0.65;
    base.leftHip.x += shift * 0.78;
    base.rightHip.x += shift * 0.78;
    base.leftKnee.x += shift * 0.82;
    base.rightKnee.x += shift * 0.82;
    base.leftAnkle.x += shift * 0.85;
    base.rightAnkle.x += shift * 0.85;

    if (reachLeft) {
      base.leftElbow.x = 0.20 + shift * 0.70;
      base.leftWrist.x = 0.04 + shift * 0.76;
      base.leftElbow.y = 0.31;
      base.leftWrist.y = 0.28;

      base.rightElbow.x = 0.66 + shift * 0.20;
      base.rightWrist.x = 0.71 + shift * 0.18;
      base.rightElbow.y = 0.44;
      base.rightWrist.y = 0.56;
    } else {
      base.rightElbow.x = 0.80 + shift * 0.70;
      base.rightWrist.x = 0.96 + shift * 0.76;
      base.rightElbow.y = 0.31;
      base.rightWrist.y = 0.28;

      base.leftElbow.x = 0.34 + shift * 0.20;
      base.leftWrist.x = 0.29 + shift * 0.18;
      base.leftElbow.y = 0.44;
      base.leftWrist.y = 0.56;
    }
  }

  return base;
}

function guidePointsToCanvas(base, w, h) {
  const marginX = 30;
  const marginY = 16;
  const usableW = w - marginX * 2;
  const usableH = h - marginY * 2;

  const out = {};
  for (const [k, p] of Object.entries(base)) {
    out[k] = {
      x: marginX + p.x * usableW,
      y: marginY + p.y * usableH
    };
  }
  return out;
}

function drawGuide(timeMs) {
  clearGuideCanvas();

  const w = guideCanvas.width;
  const h = guideCanvas.height;

  guideCtx.fillStyle = "#efefef";
  guideCtx.fillRect(0, 0, w, h);

  const task = TASKS[currentTaskIndex] || TASKS[TASKS.length - 1];
  const t = (timeMs % 1800) / 1800;
  const e = easeInOut(t < 0.5 ? t * 2 : (1 - t) * 2);

  const base = createGuidePose(task.demoType, e, timeMs);
  const pts = guidePointsToCanvas(base, w, h);

  drawSilhouette(pts);
}

function loop() {
  requestAnimationFrame(loop);

  if (!poseLandmarker) return;

  const nowMs = performance.now();
  guideTime += 16;
  drawGuide(guideTime);

  if (video.currentTime === lastVideoTime) return;

  const result = poseLandmarker.detectForVideo(video, nowMs);
  lastVideoTime = video.currentTime;

  drawResults(result);

  if (!running) return;

  if (!result.landmarks || result.landmarks.length === 0) {
    instructionBannerEl.textContent = "Step back until your full body is visible.";
    scoreValueEl.textContent = "0.0";
    return;
  }

  const landmarks = result.landmarks[0];

  if (state === "calibrating") {
    calibrationFrames.push({ landmarks });
    instructionBannerEl.textContent = "Stand still briefly for calibration.";

    if (nowMs - stateStartTime >= CALIBRATION_DURATION_MS) {
      baselinePose = computeBaselinePose(calibrationFrames);
      startDemoPhase();
    }
    return;
  }

  if (state === "demo") {
    const remain = Math.max(0, (DEMO_DURATION_MS - (nowMs - stateStartTime)) / 1000);
    instructionBannerEl.textContent = `Watch the guide first... ${remain.toFixed(1)}`;

    if (nowMs - stateStartTime >= DEMO_DURATION_MS) {
      startPerformPhase();
    }
    return;
  }

  if (state === "performing") {
    const elapsed = nowMs - stateStartTime;
    const record = getTaskRecord();

    const scored = scoreCurrentTask(landmarks, baselinePose, record);
    const rawScore = scored.score;
    smoothedScore = smoothedScore * 0.82 + rawScore * 0.18;
    scoreValueEl.textContent = smoothedScore.toFixed(1);

    record.bestPoseScore = Math.max(record.bestPoseScore, scored.poseScore);
    record.bestRawScore = Math.max(record.bestRawScore, smoothedScore);

    if (record.firstMotionAt == null && scored.motionGate > 0.40) {
      record.firstMotionAt = nowMs;
    }

    if (scored.holdReady) {
      if (record.firstReadyAt == null) {
        record.firstReadyAt = nowMs;
      }

      if (!holdStartTime) {
        holdStartTime = nowMs;
      }

      const heldMs = nowMs - holdStartTime;
      record.holdFrames.push(smoothedScore);

      const holdLeft = Math.max(0, (HOLD_DURATION_MS - heldMs) / 1000);
      instructionBannerEl.textContent = `Hold it... ${holdLeft.toFixed(1)}`;

      if (heldMs >= HOLD_DURATION_MS) {
        finalizeCurrentTask();
      }
    } else {
      holdStartTime = null;
      instructionBannerEl.textContent = TASKS[currentTaskIndex].instruction;
    }

    if (elapsed >= TASK_DURATION_MS) {
      finalizeCurrentTask();
    }
  }
}

startBtn.addEventListener("click", startTest);
resetBtn.addEventListener("click", resetTest);
modalResetBtn.addEventListener("click", () => {
  resetTest();
  startTest();
});

async function init() {
  try {
    await setupCamera();
    await createPoseLandmarker();

    setModeText();
    drawGuide(0);
    instructionBannerEl.textContent =
      "Stand where your full body is visible, then press Start Test.";
    loop();
  } catch (err) {
    console.error("INIT ERROR:", err);
    statusEl.textContent = "Error";
    instructionBannerEl.textContent = `Failed: ${err.message}`;
  }
}

init();