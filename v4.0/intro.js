import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs";

const sceneWrap = document.getElementById("sceneWrap");
const webcam = document.getElementById("webcam");
const camStatus = document.getElementById("camStatus");
const flash = document.getElementById("flash");
const subtitle = document.getElementById("subtitle");

const debugCanvas = document.getElementById("debugCanvas");
const debugCtx = debugCanvas.getContext("2d");

let scene, camera, renderer;
let particleSystem, particleGeometry, particleMaterial;

const particleCount = 22000;
let basePositions;
let velocities;
let seeds;

let poseLandmarker = null;
let lastVideoTime = -1;

let exploding = false;
let redirectScheduled = false;

let clock = new THREE.Clock();

let rightHistory = [];
let leftHistory = [];
let lastSwipeTime = 0;

let handInfluence = {
  active: false,
  x: 0,
  y: 0,
  strength: 0
};

initScene();
initParticles();
await initCameraAndPose();
animate();

function initScene() {
  scene = new THREE.Scene();

  camera = new THREE.PerspectiveCamera(
    55,
    window.innerWidth / window.innerHeight,
    0.1,
    1000
  );
  camera.position.z = 10;

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: false
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(0x000000, 1);

  sceneWrap.appendChild(renderer.domElement);

  const ambient = new THREE.AmbientLight(0xffffff, 0.8);
  scene.add(ambient);

  window.addEventListener("resize", onResize);
}

function initParticles() {
  particleGeometry = new THREE.BufferGeometry();

  const positions = new Float32Array(particleCount * 3);
  basePositions = new Float32Array(particleCount * 3);
  velocities = new Float32Array(particleCount * 3);
  seeds = new Float32Array(particleCount);

  for (let i = 0; i < particleCount; i++) {
    const i3 = i * 3;

    const u = Math.random();
    const v = Math.random();

    const theta = 2 * Math.PI * u;
    const phi = Math.acos(2 * v - 1);

    const radius = 2.8 + (Math.random() - 0.5) * 0.45;

    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.sin(phi) * Math.sin(theta);
    const z = radius * Math.cos(phi);

    positions[i3] = x;
    positions[i3 + 1] = y;
    positions[i3 + 2] = z;

    basePositions[i3] = x;
    basePositions[i3 + 1] = y;
    basePositions[i3 + 2] = z;

    velocities[i3] = 0;
    velocities[i3 + 1] = 0;
    velocities[i3 + 2] = 0;

    seeds[i] = Math.random() * Math.PI * 2;
  }

  particleGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(positions, 3)
  );

  particleMaterial = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.028,
    transparent: true,
    opacity: 0.96,
    blending: THREE.AdditiveBlending,
    depthWrite: false
  });

  particleSystem = new THREE.Points(particleGeometry, particleMaterial);
  scene.add(particleSystem);
}

async function initCameraAndPose() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: 960,
        height: 540,
        facingMode: "user"
      },
      audio: false
    });

    webcam.srcObject = stream;

    await new Promise((resolve) => {
      webcam.onloadedmetadata = () => resolve();
    });

    await webcam.play();

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

    camStatus.textContent = "CAMERA ONLINE / WAVE LEFT TO START";
  } catch (err) {
    console.error(err);
    camStatus.textContent = "CAMERA FAILED";
    subtitle.textContent = "Camera access failed";
  }
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function updateWristHistory(history, point, now) {
  history.push({
    x: point.x,
    y: point.y,
    t: now
  });

  while (history.length > 0 && now - history[0].t > 320) {
    history.shift();
  }
}

function detectLeftSwipe(history, now) {
  if (history.length < 4) return false;
  if (now - lastSwipeTime < 900) return false;

  const oldest = history[0];
  const latest = history[history.length - 1];

  // 这里用“镜像后的屏幕坐标”
  // 屏幕里向左挥 => x 变小 => dx > 0
  const dx = oldest.x - latest.x;
  const dy = Math.abs(oldest.y - latest.y);
  const dt = latest.t - oldest.t;

  const fastEnough = dt > 60 && dt < 320;
  const farEnough = dx > 0.16;
  const notTooVertical = dy < 0.14;

  if (fastEnough && farEnough && notTooVertical) {
    lastSwipeTime = now;
    return true;
  }

  return false;
}

function pickActiveHand(landmarks) {
  const leftWrist = landmarks[15];
  const rightWrist = landmarks[16];

  const leftScore = (leftWrist?.visibility ?? 0);
  const rightScore = (rightWrist?.visibility ?? 0);

  if (rightScore >= leftScore && rightScore > 0.45) {
    return { wrist: rightWrist, side: "right" };
  }

  if (leftScore > 0.45) {
    return { wrist: leftWrist, side: "left" };
  }

  return null;
}

function analyzePose() {
  if (!poseLandmarker) return;
  if (webcam.currentTime === lastVideoTime) return;

  const now = performance.now();
  const result = poseLandmarker.detectForVideo(webcam, now);
  lastVideoTime = webcam.currentTime;

  handInfluence.active = false;
  debugCtx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);

  if (!result.landmarks || !result.landmarks.length) {
    camStatus.textContent = "NO BODY DETECTED / STEP INTO FRAME";
    return;
  }

  const landmarks = result.landmarks[0];
  const activeHand = pickActiveHand(landmarks);

  if (!activeHand) {
    camStatus.textContent = "HAND NOT CLEAR / RAISE ONE HAND";
    return;
  }

  const wrist = activeHand.wrist;

  // MediaPipe 原坐标 -> 镜像屏幕坐标
  const screenX = 1 - wrist.x;
  const screenY = wrist.y;

  handInfluence.active = true;
  handInfluence.x = screenX;
  handInfluence.y = screenY;
  handInfluence.strength = 1;

  camStatus.textContent = "HAND DETECTED / SWING LEFT";

  if (activeHand.side === "right") {
    updateWristHistory(rightHistory, { x: screenX, y: screenY }, now);
    if (detectLeftSwipe(rightHistory, now)) {
      triggerExplosion();
    }
  } else {
    updateWristHistory(leftHistory, { x: screenX, y: screenY }, now);
    if (detectLeftSwipe(leftHistory, now)) {
      triggerExplosion();
    }
  }

  // 调试点，可打开 debugCanvas 看
  const dw = debugCanvas.width || 220;
  const dh = debugCanvas.height || 124;
  debugCanvas.width = 220;
  debugCanvas.height = 124;

  debugCtx.fillStyle = "rgba(255,255,255,0.08)";
  debugCtx.fillRect(0, 0, 220, 124);

  debugCtx.fillStyle = "#25ff5b";
  debugCtx.beginPath();
  debugCtx.arc(screenX * 220, screenY * 124, 6, 0, Math.PI * 2);
  debugCtx.fill();
}

function triggerExplosion() {
  if (exploding) return;

  exploding = true;
  subtitle.textContent = "Human signal accepted";
  camStatus.textContent = "SWIPE DETECTED / ENTERING";

  flash.style.opacity = "0.9";
  setTimeout(() => {
    flash.style.opacity = "0";
  }, 120);

  if (!redirectScheduled) {
    redirectScheduled = true;
    setTimeout(() => {
      window.location.href = "statement.html";
    }, 950);
  }
}

function animate() {
  requestAnimationFrame(animate);

  analyzePose();

  const positions = particleGeometry.attributes.position.array;
  const elapsed = clock.getElapsedTime();

  if (!exploding) {
    const breathe = 1 + Math.sin(elapsed * 1.4) * 0.045;

    particleSystem.rotation.y += 0.0022;
    particleSystem.rotation.x += 0.0009;
    particleSystem.rotation.z = Math.sin(elapsed * 0.5) * 0.05;

    let influenceX = 0;
    let influenceY = 0;
    let useInfluence = false;

    if (handInfluence.active) {
      influenceX = (handInfluence.x - 0.5) * 8.2;
      influenceY = (0.5 - handInfluence.y) * 5.0;
      useInfluence = true;
    }

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;

      const bx = basePositions[i3];
      const by = basePositions[i3 + 1];
      const bz = basePositions[i3 + 2];
      const seed = seeds[i];

      let x = bx * breathe;
      let y = by * breathe;
      let z = bz * breathe;

      const ripple = Math.sin(elapsed * 1.8 + seed) * 0.03;
      x += Math.cos(seed) * ripple;
      y += Math.sin(seed * 1.3) * ripple;
      z += Math.cos(seed * 1.7) * ripple;

      if (useInfluence) {
        const dx = x - influenceX;
        const dy = y - influenceY;
        const dist = Math.sqrt(dx * dx + dy * dy) + 0.001;

        if (dist < 1.6) {
          const force = (1.6 - dist) / 1.6;
          velocities[i3] += (dx / dist) * force * 0.035;
          velocities[i3 + 1] += (dy / dist) * force * 0.035;
          velocities[i3 + 2] += (Math.random() - 0.5) * force * 0.01;
        }
      }

      velocities[i3] *= 0.92;
      velocities[i3 + 1] *= 0.92;
      velocities[i3 + 2] *= 0.92;

      x += velocities[i3];
      y += velocities[i3 + 1];
      z += velocities[i3 + 2];

      positions[i3] = x;
      positions[i3 + 1] = y;
      positions[i3 + 2] = z;
    }

    particleMaterial.opacity = 0.93;
    particleMaterial.size = 0.027;
  } else {
    particleSystem.rotation.y += 0.018;
    particleSystem.rotation.x += 0.012;

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;

      const x = positions[i3];
      const y = positions[i3 + 1];
      const z = positions[i3 + 2];
      const len = Math.sqrt(x * x + y * y + z * z) + 0.0001;

      velocities[i3] += (x / len) * 0.02 + (Math.random() - 0.5) * 0.01;
      velocities[i3 + 1] += (y / len) * 0.02 + (Math.random() - 0.5) * 0.01;
      velocities[i3 + 2] += (z / len) * 0.02 + (Math.random() - 0.5) * 0.01;

      velocities[i3] *= 1.01;
      velocities[i3 + 1] *= 1.01;
      velocities[i3 + 2] *= 1.01;

      positions[i3] += velocities[i3];
      positions[i3 + 1] += velocities[i3 + 1];
      positions[i3 + 2] += velocities[i3 + 2];
    }

    particleMaterial.opacity *= 0.985;
    particleMaterial.size *= 1.003;
  }

  particleGeometry.attributes.position.needsUpdate = true;
  renderer.render(scene, camera);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}