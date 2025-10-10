export async function setupWebcam(videoEl, { width = 640, height = 480 } = {}) {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width, height }, audio: false });
  videoEl.srcObject = stream;
  await videoEl.play();
  return new Promise(res => {
    if (videoEl.readyState >= 2) return res();
    videoEl.onloadeddata = () => res();
  });
}