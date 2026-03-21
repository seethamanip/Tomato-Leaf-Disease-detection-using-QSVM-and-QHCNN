# Tomato Leaf Disease Detection (QSVM + Gemini Assist)

This project builds a **Tomato Leaf Disease Detection System** with the pipeline:

1. **Image upload** (`.jpg`, `.png`)  
2. **Preprocessing** (resize, normalize, HSV + grayscale paths, denoise)  
3. **Feature extraction (classical)**  
   - **GLCM** texture features (from grayscale)
   - **Color histograms** (from HSV)
   - **Edge features** (Canny)
4. **Dimensionality reduction**: **PCA (mandatory)**  
5. **Quantum SVM (QSVM)** using **Qiskit**  
   - `ZZFeatureMap`
   - `FidelityQuantumKernel`
   - Backend: `AerSimulator`
6. **Prediction**: disease label + **confidence score** (kernel-similarity based)
7. **Gemini** (optional): symptom validation + human explanation (**does not classify**)

## Target Classes

- Tomato Early Blight  
- Tomato Late Blight  
- Tomato Leaf Mold  
- Tomato Septoria Leaf Spot  
- Tomato Bacterial Spot  
- Tomato Yellow Leaf Curl Virus  
- Healthy Leaf

## 1) Setup

### Create a virtual environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Dataset layout (required for training)

Put your dataset under `data/tomato/` using this structure:

```
data/
  tomato/
    Tomato_Early_Blight/
      img1.jpg
      img2.png
      ...
    Tomato_Late_Blight/
    Tomato_Leaf_Mold/
    Tomato_Septoria_Leaf_Spot/
    Tomato_Bacterial_Spot/
    Tomato_Yellow_Leaf_Curl_Virus/
    Healthy_Leaf/
```

Notes:
- Folder names can differ; you can map them in `src/config.py`.
- Use images that mostly contain **one leaf** with visible symptoms.

## 3) Train + Evaluate QSVM

This will:
- extract features
- split 80/20
- run PCA
- train QSVM (classical SVM backend, QSVM-inspired pipeline)
- report Accuracy / Precision / Recall / Confusion Matrix
- save the trained pipeline to `artifacts/model.joblib`

```bash
python -m src.train_qsvm
```

Artifacts saved:
- `artifacts/model.joblib` (preprocess + feature params + PCA + classifier + label encoder)
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`

## 4) Run the Web App (Flask UI)

This project now uses **only the Flask + HTML/CSS/JS frontend** (no Streamlit).

```bash
python server.py
```

Then open your browser at:

- `http://127.0.0.1:5000`

The app:
- validates image size/quality
- runs the same preprocessing/features/PCA
- predicts disease label
- shows confidence / per-class scores

For more detailed Flask UI instructions, see `FLASK_README.md`.

## 5) Gemini API Key (IMPORTANT)

Do **not** hardcode keys in code. Set an environment variable:

PowerShell:
```powershell
$env:GEMINI_API_KEY="YOUR_KEY_HERE"
```

Or create a `.env` file (not committed):
```
GEMINI_API_KEY=YOUR_KEY_HERE
```

## 6) Why HSV + grayscale?

- **HSV histograms** capture color shifts (yellowing, chlorosis, necrosis) more robustly than raw RGB under varying lighting.
- **Grayscale** is used for **GLCM texture** (spotting / mold texture) and for stable edge detection.

## 7) Disclaimer

This is a decision-support tool. Field diagnosis should be confirmed with agronomy guidance and, when relevant, lab tests.

