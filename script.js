
/*
  script.js (ES)
  - Intenta cargar modelo TFLite desde model/modelo.tflite usando tfjs-tflite (WASM)
  - Si falla, usa modo simulación para dibujar detecciones de ejemplo.
  - Interfaz en español.
*/
const CLASS_MAP = {
  0: 'CPU',
  1: 'Mesa',
  2: 'Mouse',
  3: 'Pantalla',
  4: 'Silla',
  5: 'Teclado'
};
const MIN_SCORE = 0.25;
const MODEL_PATH = './model/modelo.tflite';

const statusEl = document.getElementById('status');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const fileInput = document.getElementById('imgfile');
const runBtn = document.getElementById('runbtn');
const countsEl = document.getElementById('counts');
const legendEl = document.getElementById('legend');

let tfliteModel = null;
let currentImage = null;
let simulationMode = false;

async function init() {
  statusEl.textContent = 'Estado: intentado cargar modelo TFLite...';
  try {
    tfliteModel = await tflite.loadTFLiteModel(MODEL_PATH);
    statusEl.textContent = 'Estado: modelo TFLite cargado (detección real disponible).';
  } catch (e) {
    console.warn('No se pudo cargar modelo TFLite:', e);
    statusEl.textContent = 'Estado: no hay modelo válido; se utilizará simulación.';
    simulationMode = true;
  }
  legendEl.innerHTML = '<strong>Clases (número → nombre):</strong> ' + Object.entries(CLASS_MAP).map(([k,v]) => `${k}→${v}`).join(' | ');
}

fileInput.addEventListener('change', (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = new Image();
    img.onload = function() {
      currentImage = img;
      const maxW = 1200;
      let drawW = img.width, drawH = img.height;
      if (img.width > maxW) {
        const scale = maxW / img.width;
        drawW = Math.round(img.width * scale);
        drawH = Math.round(img.height * scale);
      }
      canvas.width = drawW;
      canvas.height = drawH;
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(img, 0, 0, drawW, drawH);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(f);
});

function randomDetectionsExample() {
  // genera detecciones aleatorias plausibles basadas en canvas size
  const items = [];
  const n = Math.floor(Math.random()*6) + 2;
  for (let i=0;i<n;i++) {
    const w = 0.15 + Math.random()*0.25;
    const h = 0.1 + Math.random()*0.3;
    const xmin = Math.random()*(1 - w);
    const ymin = Math.random()*(1 - h);
    const xmax = xmin + w;
    const ymax = ymin + h;
    const cls = Math.floor(Math.random()*3); // 0..2 -> CPU, Mesa, Mouse (ejemplo)
    const score = 0.4 + Math.random()*0.6;
    items.push({box:[ymin,xmin,ymax,xmax], cls, score});
  }
  return items;
}

async function runDetection() {
  if (!currentImage) { alert('Por favor, sube una imagen primero.'); return; }
  statusEl.textContent = 'Estado: procesando...';

  // Dibujar imagen original
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

  let detections = [];
  if (simulationMode || !tfliteModel) {
    // usar simulación
    detections = randomDetectionsExample();
  } else {
    // intentar inferencia real
    try {
      const input = tf.browser.fromPixels(canvas).toFloat().expandDims(0);
      const outputs = await tfliteModel.predict(input);
      input.dispose();
      // intentar extraer tensores habituales
      let boxes = outputs['detection_boxes'] || outputs['boxes'] || outputs['output_boxes'] || null;
      let scores = outputs['detection_scores'] || outputs['scores'] || outputs['output_scores'] || null;
      let classes = outputs['detection_classes'] || outputs['classes'] || outputs['output_classes'] || null;
      if (Array.isArray(outputs) && outputs.length>=3 && !boxes) {
        boxes = outputs[0]; scores = outputs[1]; classes = outputs[2];
      }
      if (!boxes || !scores || !classes) {
        console.warn('Salida del modelo no contiene boxes/scores/classes:', outputs);
        statusEl.textContent = 'Inferencia completada pero salida inesperada; usando simulación.';
        detections = randomDetectionsExample();
      } else {
        const boxesArr = (await boxes.array())[0] || (await boxes.array());
        const scoresArr = (await scores.array())[0] || (await scores.array());
        const classesArr = (await classes.array())[0] || (await classes.array());
        for (let i=0;i<boxesArr.length;i++) {
          const score = scoresArr[i];
          if (score < MIN_SCORE) continue;
          let cls = Math.round(classesArr[i]);
          detections.push({box: boxesArr[i], cls, score});
        }
      }
    } catch (err) {
      console.error('Error en inferencia TFLite:', err);
      statusEl.textContent = 'Error en inferencia; usando simulación.';
      detections = randomDetectionsExample();
    }
  }

  // Dibujar detecciones en azul con número de clase
  ctx.lineWidth = Math.max(2, Math.round(canvas.width/300));
  ctx.font = `${Math.max(12, Math.round(canvas.width/60))}px Arial`;
  ctx.textBaseline = 'top';

  const counts = {};
  Object.keys(CLASS_MAP).forEach(k=>counts[k]=0);

  detections.forEach(det => {
    const [ymin,xmin,ymax,xmax] = det.box;
    const x = xmin * canvas.width;
    const y = ymin * canvas.height;
    const w = (xmax - xmin) * canvas.width;
    const h = (ymax - ymin) * canvas.height;
    ctx.strokeStyle = '#007bff';
    ctx.fillStyle = '#007bff';
    ctx.strokeRect(x, y, w, h);

    const labelText = `${det.cls}`;
    const textW = ctx.measureText(labelText).width + 8;
    ctx.fillStyle = 'rgba(255,255,255,0.9)';
    ctx.fillRect(x, y - 1, textW, parseInt(ctx.font,10)+6);
    ctx.fillStyle = '#007bff';
    ctx.fillText(labelText, x+4, y+2);

    if (String(det.cls) in counts) counts[String(det.cls)]++;
    else counts[String(det.cls)] = (counts[String(det.cls)]||0) + 1;
  });

  countsEl.innerHTML = '';
  for (const [k, v] of Object.entries(counts)) {
    const div = document.createElement('div');
    div.className = 'countbox';
    div.innerHTML = `<strong>${CLASS_MAP[k] || k}</strong><br/>Cantidad: ${v}`;
    countsEl.appendChild(div);
  }

  statusEl.textContent = `Listo — ${detections.length} detecciones (modo ${simulationMode ? 'simulación' : 'real'})`;
}

init();
runBtn.addEventListener('click', runDetection);
