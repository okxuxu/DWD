const continueBtn = document.getElementById("continueBtn");
const progressBar = document.getElementById("progressBar");

let startX = null;
let progress = 0;
let navigating = false;

function goNext(){
  if(navigating) return;
  navigating = true;
  window.location.href = "index.html";
}

function updateProgress(delta){
  const threshold = 180;
  progress = Math.max(0, Math.min(delta / threshold, 1));
  progressBar.style.width = `${progress * 100}%`;
}

function resetProgress(){
  progress = 0;
  progressBar.style.width = "0%";
}

continueBtn.addEventListener("click", goNext);

window.addEventListener("keydown", (e)=>{
  if(e.key === "Enter"){
    goNext();
  }
});

window.addEventListener("mousedown", (e)=>{
  startX = e.clientX;
});

window.addEventListener("mousemove", (e)=>{
  if(startX === null || navigating) return;
  updateProgress(startX - e.clientX);
});

window.addEventListener("mouseup", (e)=>{
  if(startX === null || navigating) return;

  const delta = startX - e.clientX;
  if(delta > 180){
    goNext();
  }else{
    resetProgress();
  }

  startX = null;
});

window.addEventListener("mouseleave", ()=>{
  startX = null;
  resetProgress();
});

window.addEventListener("touchstart", (e)=>{
  if(!e.touches.length) return;
  startX = e.touches[0].clientX;
}, { passive:true });

window.addEventListener("touchmove", (e)=>{
  if(startX === null || navigating || !e.touches.length) return;
  updateProgress(startX - e.touches[0].clientX);
}, { passive:true });

window.addEventListener("touchend", (e)=>{
  if(startX === null || navigating) return;

  const endX = e.changedTouches.length ? e.changedTouches[0].clientX : startX;
  const delta = startX - endX;

  if(delta > 180){
    goNext();
  }else{
    resetProgress();
  }

  startX = null;
});