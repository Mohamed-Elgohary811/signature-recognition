const API_ENDPOINT = "/predict";

const root = document.documentElement;
const themeToggle = document.getElementById("themeToggle");
const form = document.getElementById("predictionForm");
const input = document.getElementById("signatureInput");
const dropzone = document.getElementById("dropzone");
const previewWrap = document.getElementById("previewWrap");
const preview = document.getElementById("imagePreview");
const clearImage = document.getElementById("clearImage");
const analyzeButton = document.getElementById("analyzeButton");
const connectionStatus = document.getElementById("connectionStatus");
const emptyState = document.getElementById("emptyState");
const resultCard = document.getElementById("resultCard");
const errorBox = document.getElementById("errorBox");
const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const genuinePercent = document.getElementById("genuinePercent");
const forgedPercent = document.getElementById("forgedPercent");
const genuineBar = document.getElementById("genuineBar");
const forgedBar = document.getElementById("forgedBar");
const resultDot = document.getElementById("resultDot");
const signalCanvas = document.getElementById("signalCanvas");
const signalContext = signalCanvas.getContext("2d");

let selectedFile = null;
let previewUrl = null;
let animationFrame = null;

const savedTheme = localStorage.getItem("signature-theme");
if (savedTheme === "dark" || (!savedTheme && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
  root.dataset.theme = "dark";
}

themeToggle.addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("signature-theme", root.dataset.theme);
});

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    showError("Please choose a valid image file.");
    return;
  }

  selectedFile = file;
  analyzeButton.disabled = false;
  errorBox.hidden = true;

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }

  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  previewWrap.hidden = false;
  connectionStatus.textContent = "Image selected";
}

function resetUpload() {
  selectedFile = null;
  input.value = "";
  analyzeButton.disabled = true;
  previewWrap.hidden = true;
  connectionStatus.textContent = "Ready";

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}

function setLoading(isLoading) {
  analyzeButton.disabled = isLoading || !selectedFile;
  analyzeButton.classList.toggle("is-loading", isLoading);
  analyzeButton.querySelector(".button-label").textContent = isLoading ? "Analyzing..." : "Analyze Signature";
  connectionStatus.textContent = isLoading ? "Processing" : selectedFile ? "Image selected" : "Ready";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  emptyState.hidden = false;
  resultCard.hidden = true;
  resultDot.className = "result-dot";
}

function toPercent(value) {
  const numeric = Number(value || 0);
  const normalized = numeric <= 1 ? numeric * 100 : numeric;
  return Math.max(0, Math.min(100, normalized));
}

function renderResult(data) {
  const prediction = String(data.prediction || "Unknown");
  const predictionClass = prediction.toLowerCase().includes("forged") ? "forged" : "genuine";
  const confidence = toPercent(data.confidence);
  const genuine = toPercent(data.probabilities?.genuine);
  const forged = toPercent(data.probabilities?.forged);

  predictionText.textContent = prediction;
  predictionText.className = predictionClass;
  confidenceText.textContent = `${confidence.toFixed(1)}%`;
  genuinePercent.textContent = `${genuine.toFixed(1)}%`;
  forgedPercent.textContent = `${forged.toFixed(1)}%`;
  resultDot.className = `result-dot ${predictionClass}`;

  emptyState.hidden = true;
  resultCard.hidden = false;
  errorBox.hidden = true;

  requestAnimationFrame(() => {
    genuineBar.style.width = `${genuine}%`;
    forgedBar.style.width = `${forged}%`;
  });
}

input.addEventListener("change", (event) => {
  setFile(event.target.files[0]);
});

clearImage.addEventListener("click", resetUpload);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("drag-over");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("drag-over");
  });
});

dropzone.addEventListener("drop", (event) => {
  setFile(event.dataTransfer.files[0]);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!selectedFile) {
    showError("Please upload a signature image first.");
    return;
  }

  const formData = new FormData();
  formData.append("image_file", selectedFile);

  try {
    setLoading(true);
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Prediction failed with status ${response.status}`);
    }

    const data = await response.json();
    renderResult(data);
    connectionStatus.textContent = "Complete";
  } catch (error) {
    showError(error.message || "Prediction failed. Please try another image.");
    connectionStatus.textContent = "Error";
  } finally {
    setLoading(false);
  }
});

function drawSignalBackground(time = 0) {
  const width = signalCanvas.clientWidth;
  const height = signalCanvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;

  if (signalCanvas.width !== Math.floor(width * ratio) || signalCanvas.height !== Math.floor(height * ratio)) {
    signalCanvas.width = Math.floor(width * ratio);
    signalCanvas.height = Math.floor(height * ratio);
    signalContext.setTransform(ratio, 0, 0, ratio, 0, 0);
  }

  signalContext.clearRect(0, 0, width, height);

  const isDark = root.dataset.theme === "dark";
  signalContext.globalAlpha = isDark ? 0.38 : 0.48;
  signalContext.strokeStyle = isDark ? "#35c49a" : "#0f8f72";
  signalContext.lineWidth = 2;

  for (let row = 0; row < 7; row += 1) {
    const yBase = height * (0.22 + row * 0.095);
    signalContext.beginPath();

    for (let x = -20; x <= width + 20; x += 12) {
      const wave = Math.sin((x * 0.012) + time * 0.0012 + row) * (18 + row * 3);
      const fine = Math.cos((x * 0.031) + time * 0.0008) * 6;
      const y = yBase + wave + fine;

      if (x === -20) {
        signalContext.moveTo(x, y);
      } else {
        signalContext.lineTo(x, y);
      }
    }

    signalContext.stroke();
  }

  signalContext.globalAlpha = isDark ? 0.24 : 0.18;
  signalContext.fillStyle = isDark ? "#f0b35b" : "#d98a1b";

  for (let i = 0; i < 34; i += 1) {
    const x = (i * 137 + time * 0.018) % (width + 80) - 40;
    const y = (height * 0.16) + ((i * 53) % Math.max(1, height * 0.62));
    signalContext.beginPath();
    signalContext.arc(x, y, 2.2, 0, Math.PI * 2);
    signalContext.fill();
  }

  animationFrame = requestAnimationFrame(drawSignalBackground);
}

animationFrame = requestAnimationFrame(drawSignalBackground);

window.addEventListener("beforeunload", () => {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  cancelAnimationFrame(animationFrame);
});
