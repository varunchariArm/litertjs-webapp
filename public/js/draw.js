export function drawTopK(canvas, topK) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "16px sans-serif";
  ctx.fillStyle = "white";
  ctx.textBaseline = "top";
  topK.forEach((item, i) => {
    ctx.fillText(`${item.label}: ${(item.prob * 100).toFixed(1)}%`, 10, 10 + i * 20);
  });
}

export function resizeCanvasToVideo(canvas, video) {
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

export function drawSegmentationOverlay(canvas, video, rgba, w, h, alpha = 0.5) {
  const ctx = canvas.getContext('2d');
  // base video frame
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  // reuse a cached offscreen canvas for the mask to reduce GC / allocations
  let off = canvas._maskCanvas;
  if (!off) {
    off = document.createElement('canvas');
    canvas._maskCanvas = off;
  }
  if (off.width !== w || off.height !== h) {
    off.width = w;
    off.height = h;
  }

  const ictx = off.getContext('2d');
  const img = new ImageData(rgba, w, h);
  ictx.putImageData(img, 0, 0);

  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  ctx.restore();
}