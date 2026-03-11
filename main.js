import { FilesetResolver, PoseLandmarker } from "@mediapipe/tasks-vision";
import * as THREE from "three";
const successOverlay = document.getElementById("successOverlay");
const successProbability = document.getElementById("successProbability");
const successTimeText = document.getElementById("successTimeText");
const restartTestBtn = document.getElementById("restartTestBtn");
const video = document.getElementById("camera");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const avatarContainer = document.getElementById("avatar3d");

const secondScoreEl = document.getElementById("secondScore");
const overallScoreEl = document.getElementById("overallScore");
const realnessScoreEl = document.getElementById("realnessScore");
const statusText = document.getElementById("statusText");
const startTestBtn = document.getElementById("startTestBtn");

let poseLandmarker = null;
let lastVideoTime = -1;
let startTime = 0;

let referenceMotion = null;
let smoothedUserSkeleton = null;

let testStarted = false;
let testPassed = false;
let testStartTimestamp = 0;

const START_DELAY_SEC = 5.0;
const SCORE_TIME_WINDOW_SEC = 0.6;
const SCORE_TIME_STEP_SEC = 0.1;
const VISIBILITY_THRESHOLD = 0.25;
const SMOOTH_ALPHA = 0.65;
const DISPLAY_SCORE_MULTIPLIER = 2.8;
const PASS_THRESHOLD = 70;
const MIN_TEST_DURATION_SEC = 10;

const IDX = {
  LEFT_SHOULDER: 11,
  RIGHT_SHOULDER: 12,
  LEFT_ELBOW: 13,
  RIGHT_ELBOW: 14,
  LEFT_WRIST: 15,
  RIGHT_WRIST: 16,
  LEFT_HIP: 23,
  RIGHT_HIP: 24,
  LEFT_KNEE: 25,
  RIGHT_KNEE: 26,
  LEFT_ANKLE: 27,
  RIGHT_ANKLE: 28
};

const REQUIRED_MP = [
  11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
];

const USER_BONES = [
  [11, 12], [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 23], [12, 24],
  [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28]
];

const REF_BONES = [
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
];

let currentSecondIndex = -1;
let currentSecondFrameScores = [];
let completedSecondScores = [];
let currentSecondAvg = 0;
let cumulativeScore = 0;

let scene, camera3D, renderer;
let jointMeshes = {};
let lineGeometry, linePositions, lineSegments;

function showSuccessOverlay(probabilityValue, elapsedSec) {
  successProbability.textContent = `${Math.round(probabilityValue)}%`;
  successTimeText.textContent =
    `You wasted ${elapsedSec.toFixed(1)} seconds proving that you are human.`;

  successOverlay.classList.remove("hidden");
}

function hideSuccessOverlay() {
  successOverlay.classList.add("hidden");
}

function average(arr) {
  if (!arr.length) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function midpoint(a, b) {
  return [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dist(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

function norm(v) {
  const n = Math.hypot(v[0], v[1], v[2]);
  if (n < 1e-6) return [0, 0, 0];
  return [v[0] / n, v[1] / n, v[2] / n];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function angleBetween(a, b) {
  const na = norm(a);
  const nb = norm(b);
  const c = Math.max(-1, Math.min(1, dot(na, nb)));
  return Math.acos(c) * 180 / Math.PI;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function lerpVec3(a, b, t) {
  return [
    lerp(a[0], b[0], t),
    lerp(a[1], b[1], t),
    lerp(a[2], b[2], t)
  ];
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function setStatus(text) {
  statusText.textContent = text;
}

function normalizeSkeleton(sk) {
  const hipCenter = midpoint(sk.left_hip, sk.right_hip);
  const shoulderCenter = midpoint(sk.left_shoulder, sk.right_shoulder);
  const scale = dist(hipCenter, shoulderCenter) || 1.0;

  const out = {};
  for (const key in sk) {
    out[key] = [
      (sk[key][0] - hipCenter[0]) / scale,
      (sk[key][1] - hipCenter[1]) / scale,
      (sk[key][2] - hipCenter[2]) / scale
    ];
  }
  return out;
}

function skeletonVectors(sk) {
  return {
    lUpperArm: sub(sk.left_elbow, sk.left_shoulder),
    lLowerArm: sub(sk.left_wrist, sk.left_elbow),
    rUpperArm: sub(sk.right_elbow, sk.right_shoulder),
    rLowerArm: sub(sk.right_wrist, sk.right_elbow),
    lUpperLeg: sub(sk.left_knee, sk.left_hip),
    lLowerLeg: sub(sk.left_ankle, sk.left_knee),
    rUpperLeg: sub(sk.right_knee, sk.right_hip),
    rLowerLeg: sub(sk.right_ankle, sk.right_knee),
    torso: sub(
      midpoint(sk.left_shoulder, sk.right_shoulder),
      midpoint(sk.left_hip, sk.right_hip)
    ),
    shoulderLine: sub(sk.right_shoulder, sk.left_shoulder),
    hipLine: sub(sk.right_hip, sk.left_hip)
  };
}

function compareSkeletons(userSk, refSk) {
  const u = skeletonVectors(normalizeSkeleton(userSk));
  const r = skeletonVectors(normalizeSkeleton(refSk));

  const weightedParts = [
    ["lUpperArm", 2.2],
    ["lLowerArm", 2.2],
    ["rUpperArm", 2.2],
    ["rLowerArm", 2.2],
    ["torso", 1.8],
    ["shoulderLine", 1.5],
    ["hipLine", 1.0],
    ["lUpperLeg", 1.0],
    ["lLowerLeg", 0.9],
    ["rUpperLeg", 1.0],
    ["rLowerLeg", 0.9]
  ];

  let weightedErrorSum = 0;
  let weightSum = 0;

  for (const [key, weight] of weightedParts) {
    const err = angleBetween(u[key], r[key]);
    weightedErrorSum += err * weight;
    weightSum += weight;
  }

  const avgError = weightSum > 0 ? weightedErrorSum / weightSum : 180;

  let score = 100 * (1 - avgError / 100);

  const shoulderErr = angleBetween(u.shoulderLine, r.shoulderLine);
  const torsoErr = angleBetween(u.torso, r.torso);
  const poseBonus = Math.max(0, 8 - 0.08 * (shoulderErr + torsoErr));
  score += poseBonus;

  return clamp(score, 0, 100);
}

function smoothSkeleton(current, previous, alpha = SMOOTH_ALPHA) {
  if (!previous) return current;

  const out = {};
  for (const key in current) {
    out[key] = [
      alpha * current[key][0] + (1 - alpha) * previous[key][0],
      alpha * current[key][1] + (1 - alpha) * previous[key][1],
      alpha * current[key][2] + (1 - alpha) * previous[key][2]
    ];
  }
  return out;
}

function extractUserSkeletonFromWorld(worldLandmarks) {
  const p = (i) => {
    const lm = worldLandmarks[i];
    return [lm.x, lm.y, lm.z ?? 0];
  };

  return {
    left_shoulder: p(IDX.LEFT_SHOULDER),
    right_shoulder: p(IDX.RIGHT_SHOULDER),
    left_elbow: p(IDX.LEFT_ELBOW),
    right_elbow: p(IDX.RIGHT_ELBOW),
    left_wrist: p(IDX.LEFT_WRIST),
    right_wrist: p(IDX.RIGHT_WRIST),
    left_hip: p(IDX.LEFT_HIP),
    right_hip: p(IDX.RIGHT_HIP),
    left_knee: p(IDX.LEFT_KNEE),
    right_knee: p(IDX.RIGHT_KNEE),
    left_ankle: p(IDX.LEFT_ANKLE),
    right_ankle: p(IDX.RIGHT_ANKLE)
  };
}

function minVisibility(landmarks) {
  let m = 1;
  for (const idx of REQUIRED_MP) {
    const v = landmarks[idx]?.visibility ?? 1;
    m = Math.min(m, v);
  }
  return m;
}

function drawUserOverlay(landmarks) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  function map(lm) {
    return [
      overlay.width - lm.x * overlay.width,
      lm.y * overlay.height
    ];
  }

  overlayCtx.lineWidth = 3;
  overlayCtx.strokeStyle = "#ffffff";
  overlayCtx.fillStyle = "#ffffff";

  for (const [a, b] of USER_BONES) {
    const p1 = map(landmarks[a]);
    const p2 = map(landmarks[b]);

    overlayCtx.beginPath();
    overlayCtx.moveTo(p1[0], p1[1]);
    overlayCtx.lineTo(p2[0], p2[1]);
    overlayCtx.stroke();
  }

  for (const idx of REQUIRED_MP) {
    const p = map(landmarks[idx]);
    overlayCtx.beginPath();
    overlayCtx.arc(p[0], p[1], 5, 0, Math.PI * 2);
    overlayCtx.fill();
  }
}

function updateSecondAndCumulativeScore(frameScore, elapsedSec) {
  const sec = Math.floor(elapsedSec);

  if (currentSecondIndex === -1) {
    currentSecondIndex = sec;
  }

  if (sec !== currentSecondIndex) {
    if (currentSecondFrameScores.length > 0) {
      const finishedSecondAvg = average(currentSecondFrameScores);
      completedSecondScores.push(finishedSecondAvg);
      cumulativeScore = average(completedSecondScores);
    }
    currentSecondFrameScores = [];
    currentSecondIndex = sec;
  }

  currentSecondFrameScores.push(frameScore);
  currentSecondAvg = average(currentSecondFrameScores);
}

function scoreClass(score) {
  if (score >= 80) return "good";
  if (score >= 55) return "warn";
  return "bad";
}

function handlePassIfNeeded(realnessValue) {
  if (!testStarted || testPassed) return;

  const elapsedSec = Math.max(0, (performance.now() - startTime) / 1000);

  if (elapsedSec < MIN_TEST_DURATION_SEC) {
    return;
  }

  if (realnessValue <= PASS_THRESHOLD) {
    return;
  }

  testPassed = true;

  showSuccessOverlay(realnessValue, elapsedSec);

  setStatus("Test passed");
  startTestBtn.disabled = false;
  startTestBtn.textContent = "Start Again";
  testStarted = false;
}
function refreshScoreUI() {
  const secondDisplay = Math.min(100, currentSecondAvg * DISPLAY_SCORE_MULTIPLIER);

  const overallRaw = completedSecondScores.length > 0
    ? cumulativeScore
    : currentSecondAvg;

  const overallDisplay = Math.min(100, overallRaw * DISPLAY_SCORE_MULTIPLIER);

  secondScoreEl.textContent = `${secondDisplay.toFixed(0)}%`;
  overallScoreEl.textContent = `${overallDisplay.toFixed(0)}%`;
  realnessScoreEl.textContent = `${overallDisplay.toFixed(0)}%`;

  secondScoreEl.className = scoreClass(secondDisplay);
  overallScoreEl.className = scoreClass(overallDisplay);
  realnessScoreEl.className = scoreClass(overallDisplay);

  handlePassIfNeeded(overallDisplay);
}

function getInterpolatedReferenceFrame(tSec) {
  if (!referenceMotion || !referenceMotion.frames?.length) return null;

  const frames = referenceMotion.frames;
  const duration = frames[frames.length - 1].t || 1;
  const loopTime = ((tSec % duration) + duration) % duration;

  let i = 0;
  while (i < frames.length - 1 && frames[i + 1].t < loopTime) {
    i++;
  }

  const a = frames[i];
  const b = frames[(i + 1) % frames.length];

  const dt = (b.t >= a.t) ? (b.t - a.t) : (duration - a.t + b.t);
  const local = (loopTime - a.t >= 0) ? (loopTime - a.t) : (duration - a.t + loopTime);
  const t = dt < 1e-6 ? 0 : local / dt;

  const out = {};
  for (const key in a.joints) {
    out[key] = lerpVec3(a.joints[key], b.joints[key], t);
  }
  return out;
}

function findBestReferenceMatch(userSk, elapsedSec) {
  let bestScore = -Infinity;

  for (
    let offset = -SCORE_TIME_WINDOW_SEC;
    offset <= SCORE_TIME_WINDOW_SEC + 1e-6;
    offset += SCORE_TIME_STEP_SEC
  ) {
    const ref = getInterpolatedReferenceFrame(elapsedSec + offset);
    if (!ref) continue;

    const score = compareSkeletons(userSk, ref);
    if (score > bestScore) {
      bestScore = score;
    }
  }

  return bestScore > -Infinity ? bestScore : 0;
}

function initThree() {
  scene = new THREE.Scene();

  const width = avatarContainer.clientWidth;
  const height = avatarContainer.clientHeight;

  camera3D = new THREE.PerspectiveCamera(35, width / height, 0.1, 100);
  camera3D.position.set(0, 0, 4.8);
  camera3D.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(width, height);
  avatarContainer.appendChild(renderer.domElement);

  const ambient = new THREE.AmbientLight(0xffffff, 1.0);
  scene.add(ambient);

  const jointMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const jointGeometry = new THREE.SphereGeometry(0.05, 16, 16);

  const jointNames = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
  ];

  jointNames.forEach((name) => {
    const mesh = new THREE.Mesh(jointGeometry, jointMaterial);
    jointMeshes[name] = mesh;
    scene.add(mesh);
  });

  linePositions = new Float32Array(REF_BONES.length * 2 * 3);
  lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));

  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
  lineSegments = new THREE.LineSegments(lineGeometry, lineMaterial);
  scene.add(lineSegments);

  window.addEventListener("resize", () => {
    const w = avatarContainer.clientWidth;
    const h = avatarContainer.clientHeight;
    renderer.setSize(w, h);
    camera3D.aspect = w / h;
    camera3D.updateProjectionMatrix();
    resizeCanvas();
  });
}

function toThreeSpace(p) {
  return new THREE.Vector3(
    p[0] * 1.2,
    -p[1] * 1.2,
    -p[2] * 1.2
  );
}

function update3DAvatar(refSk) {
  if (!refSk) return;

  const sk = normalizeSkeleton(refSk);

  Object.keys(jointMeshes).forEach((name) => {
    jointMeshes[name].position.copy(toThreeSpace(sk[name]));
  });

  let ptr = 0;
  for (const [a, b] of REF_BONES) {
    const pa = toThreeSpace(sk[a]);
    const pb = toThreeSpace(sk[b]);

    linePositions[ptr++] = pa.x;
    linePositions[ptr++] = pa.y;
    linePositions[ptr++] = pa.z;

    linePositions[ptr++] = pb.x;
    linePositions[ptr++] = pb.y;
    linePositions[ptr++] = pb.z;
  }

  lineGeometry.attributes.position.needsUpdate = true;
}

function renderThree() {
  if (renderer && scene && camera3D) {
    renderer.render(scene, camera3D);
  }
}

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
  await video.play();
  resizeCanvas();
  window.addEventListener("resize", resizeCanvas);
}

function resizeCanvas() {
  overlay.width = window.innerWidth;
  overlay.height = window.innerHeight;
}

async function setupPose() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "/assets/pose_landmarker_lite.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });
}

async function loadReferenceMotion() {
  const response = await fetch("/reference_motion.json");
  referenceMotion = await response.json();
}

function resetScores() {
  currentSecondIndex = -1;
  currentSecondFrameScores = [];
  completedSecondScores = [];
  currentSecondAvg = 0;
  cumulativeScore = 0;
  refreshScoreUI();
}

function startTest() {
  if (testStarted) return;

  hideSuccessOverlay();

  testStarted = true;
  testPassed = false;
  testStartTimestamp = performance.now();
  startTime = performance.now() + START_DELAY_SEC * 1000;

  resetScores();
  smoothedUserSkeleton = null;

  startTestBtn.disabled = true;
  setStatus("Test started");
}

async function loop() {
  const nowMs = performance.now();

  if (!testStarted) {
    const idleRef = getInterpolatedReferenceFrame(0);
    update3DAvatar(idleRef);
    renderThree();
    requestAnimationFrame(loop);
    return;
  }

  const elapsedSecRaw = (nowMs - startTime) / 1000;
  const elapsedSec = Math.max(0, elapsedSecRaw);

  const refFrameForAvatar = getInterpolatedReferenceFrame(elapsedSec);
  update3DAvatar(refFrameForAvatar);
  renderThree();

  if (elapsedSecRaw < 0) {
    const remain = Math.ceil(-elapsedSecRaw);
    setStatus(`Get ready... ${remain}`);
    requestAnimationFrame(loop);
    return;
  }

  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    const result = poseLandmarker.detectForVideo(video, nowMs);

    if (result.landmarks?.length > 0) {
      const imgLandmarks = result.landmarks[0];
      drawUserOverlay(imgLandmarks);

      const vis = minVisibility(imgLandmarks);
      if (vis < VISIBILITY_THRESHOLD) {
        setStatus("Move back slightly so your full body is visible.");
        requestAnimationFrame(loop);
        return;
      }

      if (result.worldLandmarks?.length > 0) {
        const worldLandmarks = result.worldLandmarks[0];
        const rawUserSk = extractUserSkeletonFromWorld(worldLandmarks);

        smoothedUserSkeleton = smoothSkeleton(rawUserSk, smoothedUserSkeleton, SMOOTH_ALPHA);

        const frameScore = findBestReferenceMatch(smoothedUserSkeleton, elapsedSec);

        updateSecondAndCumulativeScore(frameScore, elapsedSec);
        refreshScoreUI();
        setStatus("Tracking");
      }
    } else {
      overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
      setStatus("No person detected");
    }
  }

  requestAnimationFrame(loop);
}

async function init() {
  try {
    setStatus("Loading camera...");
    await setupCamera();

    setStatus("Loading pose model...");
    await setupPose();

    setStatus("Loading reference motion...");
    await loadReferenceMotion();

    initThree();
    resetScores();

    smoothedUserSkeleton = null;
    setStatus("Click Start Test to begin");

    startTestBtn.addEventListener("click", startTest);
    restartTestBtn.addEventListener("click", startTest);


    

    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    setStatus(`Failed: ${err.message}`);
  }
}

init();