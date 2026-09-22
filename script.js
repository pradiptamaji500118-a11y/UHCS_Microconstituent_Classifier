const MODEL_PATH = "microconstituent_resnet34.onnx";
const LABELS_PATH = "label_classes.json";
const IMG_SIZE = 224;
const RESIZE_SHORT_SIDE = Math.round(IMG_SIZE * 1.15); // matches training's eval_transform
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let session = null;
let classNames = [];
let currentImage = null;

const fileInput = document.getElementById("file-input");
const dropLabel = document.getElementById("drop-label");
const preview = document.getElementById("preview");
const analyzeBtn = document.getElementById("analyze-btn");
const statusEl = document.getElementById("status");
const resultsCard = document.getElementById("results-card");
const resultsList = document.getElementById("results-list");

async function init() {
  try {
    const [loadedSession, labels] = await Promise.all([
      ort.InferenceSession.create(MODEL_PATH),
      fetch(LABELS_PATH).then((r) => r.json()),
    ]);
    session = loadedSession;
    classNames = labels;
    statusEl.textContent = "Model ready. Upload a micrograph.";
  } catch (err) {
    statusEl.textContent = "Failed to load model: " + err.message;
    console.error(err);
  }
}
init();

function handleFile(file) {
  if (!file) return;
  const img = new Image();
  img.onload = () => {
    currentImage = img;
    preview.src = img.src;
    preview.hidden = false;
    analyzeBtn.disabled = false;
    resultsCard.hidden = true;
  };
  img.src = URL.createObjectURL(file);
}

fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragover", "drop"].forEach((evt) =>
  dropLabel.addEventListener(evt, (e) => e.preventDefault())
);
dropLabel.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

function preprocess(img) {
  const scale = RESIZE_SHORT_SIDE / Math.min(img.width, img.height);
  const resizedW = Math.round(img.width * scale);
  const resizedH = Math.round(img.height * scale);

  const resizeCanvas = document.createElement("canvas");
  resizeCanvas.width = resizedW;
  resizeCanvas.height = resizedH;
  resizeCanvas.getContext("2d").drawImage(img, 0, 0, resizedW, resizedH);

  // Center crop to IMG_SIZE x IMG_SIZE (matches transforms.CenterCrop).
  const cropX = Math.floor((resizedW - IMG_SIZE) / 2);
  const cropY = Math.floor((resizedH - IMG_SIZE) / 2);

  const cropCanvas = document.createElement("canvas");
  cropCanvas.width = IMG_SIZE;
  cropCanvas.height = IMG_SIZE;
  cropCanvas
    .getContext("2d")
    .drawImage(resizeCanvas, cropX, cropY, IMG_SIZE, IMG_SIZE, 0, 0, IMG_SIZE, IMG_SIZE);

  const { data } = cropCanvas.getContext("2d").getImageData(0, 0, IMG_SIZE, IMG_SIZE);

  // RGBA -> normalized CHW Float32Array (matches ToTensor + Normalize).
  const plane = IMG_SIZE * IMG_SIZE;
  const chw = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = (r - MEAN[0]) / STD[0];
    chw[plane + i] = (g - MEAN[1]) / STD[1];
    chw[2 * plane + i] = (b - MEAN[2]) / STD[2];
  }
  return new ort.Tensor("float32", chw, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

analyzeBtn.addEventListener("click", async () => {
  if (!currentImage || !session) return;
  statusEl.textContent = "Analyzing...";
  analyzeBtn.disabled = true;
  try {
    const inputTensor = preprocess(currentImage);
    const outputs = await session.run({ input: inputTensor });
    const logits = Array.from(outputs.logits.data);
    const probs = softmax(logits);

    const ranked = classNames
      .map((name, i) => ({ name, prob: probs[i] }))
      .sort((a, b) => b.prob - a.prob);

    resultsList.innerHTML = "";
    ranked.forEach(({ name, prob }) => {
      const row = document.createElement("div");
      row.className = "result-row";
      row.innerHTML = `
        <span class="result-name">${name}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(prob * 100).toFixed(1)}%"></div></div>
        <span class="result-pct">${(prob * 100).toFixed(1)}%</span>
      `;
      resultsList.appendChild(row);
    });

    resultsCard.hidden = false;
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
    statusEl.textContent = "";
  } catch (err) {
    statusEl.textContent = "Error running model: " + err.message;
    console.error(err);
  } finally {
    analyzeBtn.disabled = false;
  }
});
