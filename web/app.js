const fileEl = document.getElementById("file");
const btnEl = document.getElementById("btn");
const previewEl = document.getElementById("preview");
const statusEl = document.getElementById("status");
const resultQsvmEl = document.getElementById("result_qsvm");
const resultQhcnnEl = document.getElementById("result_qhcnn");
const resultEnsembleEl = document.getElementById("result_ensemble");
const stepsQsvmEl = document.getElementById("steps_qsvm");
const stepsQhcnnEl = document.getElementById("steps_qhcnn");
const datasetInfoEl = document.getElementById("dataset_info");

function setStatus(text, kind = "muted") {
  statusEl.textContent = text;
  statusEl.className = `status ${kind}`;
}

function clearSteps() {
  stepsQsvmEl.innerHTML = "";
  stepsQhcnnEl.innerHTML = "";
}

function renderSteps(targetEl, steps) {
  targetEl.innerHTML = "";
  for (const s of steps || []) {
    const wrap = document.createElement("div");
    wrap.className = "step";

    const title = document.createElement("div");
    title.className = "title";
    title.textContent = s.title || "Step";

    const img = document.createElement("img");
    img.alt = s.title || "step";
    img.src = s.image;

    wrap.appendChild(title);
    wrap.appendChild(img);
    targetEl.appendChild(wrap);
  }
}

function kv(k, v) {
  const row = document.createElement("div");
  row.className = "kv";
  const kEl = document.createElement("div");
  kEl.className = "k";
  kEl.textContent = k;
  const vEl = document.createElement("div");
  vEl.className = "v";
  vEl.textContent = v;
  row.appendChild(kEl);
  row.appendChild(vEl);
  return row;
}

function clearResults() {
  resultQsvmEl.innerHTML = "";
  resultQhcnnEl.innerHTML = "";
  if (resultEnsembleEl) resultEnsembleEl.innerHTML = "";
  if (datasetInfoEl) datasetInfoEl.innerHTML = "";
}

// ── Show saved performance metrics on page load ───────────────────────────────

const PERF_METRICS = [
  { label: "Accuracy", key: "accuracy", fmt: v => (v * 100).toFixed(2) + "%" },
  { label: "Precision (weighted)", key: "precision", fmt: v => (v * 100).toFixed(2) + "%" },
  { label: "Recall (weighted)", key: "recall", fmt: v => (v * 100).toFixed(2) + "%" },
  { label: "F1 Score (weighted)", key: "f1", fmt: v => (v * 100).toFixed(2) + "%" },
];

function renderMetrics(targetEl, m) {
  targetEl.innerHTML = "";
  if (!m) return;
  for (const { label, key, fmt } of PERF_METRICS) {
    if (m[key] !== undefined && m[key] !== null) {
      targetEl.appendChild(kv(label, fmt(m[key])));
    }
  }
}



function renderModelResult(targetEl, name, valid, pred, validityScores, perf) {
  targetEl.innerHTML = "";
  if (!valid) {
    targetEl.appendChild(kv("Valid", "No"));
    if (validityScores) {
      if (validityScores.confidence !== undefined) {
        targetEl.appendChild(kv("Confidence", String(validityScores.confidence)));
      }
    }
    return;
  }
  if (!pred) return;
  targetEl.appendChild(kv("Valid", "Yes"));
  targetEl.appendChild(kv("Predicted disease", pred.disease_name));
  targetEl.appendChild(kv("Confidence", Number(pred.confidence_score).toFixed(3)));
  if (pred.accuracy_percent !== null && pred.accuracy_percent === pred.accuracy_percent) {
    targetEl.appendChild(kv("Accuracy (test set)", `${Number(pred.accuracy_percent).toFixed(2)}%`));
  }
  if (perf) {
    if (perf.precision !== undefined) {
      targetEl.appendChild(kv("Precision (weighted)", Number(perf.precision * 100).toFixed(2) + "%"));
    }
    if (perf.recall !== undefined) {
      targetEl.appendChild(kv("Recall (weighted)", Number(perf.recall * 100).toFixed(2) + "%"));
    }
    if (perf.f1 !== undefined) {
      targetEl.appendChild(kv("F1 (weighted)", Number(perf.f1 * 100).toFixed(2) + "%"));
    }
    // Also surface dataset usage per model inside the card itself.
    if (perf.num_samples !== undefined) {
      targetEl.appendChild(kv(`${name} - total images`, String(perf.num_samples)));
    }
    if (perf.num_train_samples !== undefined) {
      let txt = String(perf.num_train_samples);
      if (perf.train_fraction !== undefined) {
        txt += ` (${Number(perf.train_fraction * 100).toFixed(1)}% train)`;
      }
      targetEl.appendChild(kv(`${name} - training images`, txt));
    }
    if (perf.num_test_samples !== undefined) {
      let txt = String(perf.num_test_samples);
      if (perf.test_fraction !== undefined) {
        txt += ` (${Number(perf.test_fraction * 100).toFixed(1)}% test)`;
      }
      targetEl.appendChild(kv(`${name} - test images`, txt));
    }
  }
}

function renderResult(payload) {
  clearResults();
  if (!payload.ok) {
    setStatus(payload.error || "Request failed.", "bad");
    return;
  }

  if (!payload.valid) {
    setStatus(payload.invalid_reason || "Image is not valid.", "bad");
    renderModelResult(resultQsvmEl, "QSVM", false, null, payload.validity_scores, payload.metrics?.qsvm);
    renderModelResult(resultQhcnnEl, "QHCNN", false, null, payload.validity_scores, payload.metrics?.qhcnn);
    if (resultEnsembleEl) renderModelResult(resultEnsembleEl, "Ensemble", false, null, payload.validity_scores, null);
    return;
  }

  setStatus("Prediction completed.", "ok");
  const qsvm = payload.predictions?.qsvm;
  const qhcnn = payload.predictions?.qhcnn;
  const ensemble = payload.predictions?.ensemble;
  
  renderModelResult(resultQsvmEl, "QSVM", true, qsvm, payload.validity_scores, payload.metrics?.qsvm);
  renderModelResult(resultQhcnnEl, "QHCNN", true, qhcnn, payload.validity_scores, payload.metrics?.qhcnn);
  
  if (resultEnsembleEl && ensemble) {
     renderModelResult(resultEnsembleEl, "Optimized Ensemble", true, ensemble, payload.validity_scores, null);
     // append weights info
     if (ensemble.weights) {
         resultEnsembleEl.appendChild(kv("Weight (QSVM)", Number(ensemble.weights.qsvm).toFixed(3)));
         resultEnsembleEl.appendChild(kv("Weight (QHCNN)", Number(ensemble.weights.qhcnn).toFixed(3)));
     }
  }

  // Show dataset usage if metrics are available
  const perfQsvm = payload.metrics?.qsvm;
  const perfQhcnn = payload.metrics?.qhcnn;
  if (datasetInfoEl && (perfQsvm || perfQhcnn)) {
    const frag = document.createDocumentFragment();

    if (perfQsvm) {
      frag.appendChild(kv("QSVM - total images", String(perfQsvm.num_samples ?? "n/a")));
      if (perfQsvm.num_train_samples !== undefined) {
        let txt = String(perfQsvm.num_train_samples);
        if (perfQsvm.train_fraction !== undefined) {
          txt += ` (${Number(perfQsvm.train_fraction * 100).toFixed(1)}% train)`;
        }
        frag.appendChild(kv("QSVM - training images", txt));
      }
      if (perfQsvm.num_test_samples !== undefined) {
        let txt = String(perfQsvm.num_test_samples);
        if (perfQsvm.test_fraction !== undefined) {
          txt += ` (${Number(perfQsvm.test_fraction * 100).toFixed(1)}% test)`;
        }
        frag.appendChild(kv("QSVM - test images", txt));
      }
    }

    if (perfQhcnn) {
      // QHCNN metrics may not have explicit train/test counts; show total only.
      if (perfQhcnn.num_samples !== undefined) {
        frag.appendChild(kv("QHCNN - total images", String(perfQhcnn.num_samples)));
      }
    }

    datasetInfoEl.innerHTML = "";
    datasetInfoEl.appendChild(frag);
  }
}

fileEl.addEventListener("change", () => {
  const f = fileEl.files?.[0];
  btnEl.disabled = !f;
  clearResults();
  clearSteps();

  if (!f) {
    previewEl.classList.add("empty");
    previewEl.textContent = "No image selected";
    setStatus("Waiting for an image…", "muted");
    return;
  }

  const url = URL.createObjectURL(f);
  previewEl.classList.remove("empty");
  previewEl.innerHTML = "";
  const img = document.createElement("img");
  img.src = url;
  img.alt = "upload preview";
  previewEl.appendChild(img);
  setStatus("Ready. Click Predict.", "muted");
});

btnEl.addEventListener("click", async () => {
  const f = fileEl.files?.[0];
  if (!f) return;

  setStatus("Uploading…", "muted");
  clearResults();
  clearSteps();

  const fd = new FormData();
  fd.append("image", f);

  try {
    const resp = await fetch("/api/predict", { method: "POST", body: fd });
    const payload = await resp.json();
    renderResult(payload);
    renderSteps(stepsQsvmEl, payload.steps?.qsvm);
    renderSteps(stepsQhcnnEl, payload.steps?.qhcnn);
  } catch (e) {
    setStatus(`Request failed: ${e}`, "bad");
  }
});

