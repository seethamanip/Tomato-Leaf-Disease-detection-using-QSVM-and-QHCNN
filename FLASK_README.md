# 🍅 Tomato Leaf Disease Detection - HTML/CSS/JavaScript Frontend

This is a custom HTML/CSS/JavaScript frontend for the Tomato Leaf Disease Detection system, replacing Streamlit with a Flask backend API.

## 📁 Project Structure

```
placement/
├── server.py                 # Flask backend server
├── static/
│   ├── index.html           # Main HTML page
│   ├── style.css            # Styling
│   └── script.js            # JavaScript logic
├── src/                     # Model & inference code
├── artifacts/
│   └── model.joblib         # Trained QSVM model
└── data/tomato/             # Training dataset
```

## 🚀 How to Run

### Step 1: Open PowerShell
Press `Win + R`, type `powershell`, press Enter

### Step 2: Navigate to Project
```powershell
cd c:\Users\Admin\Desktop\placement
```

### Step 3: Activate Virtual Environment
```powershell
.venv\Scripts\Activate
```
You should see `(.venv)` in your command prompt

### Step 4: Install Flask (if not already installed)
```powershell
pip install flask flask-cors
```

### Step 5: Run the Server
```powershell
python server.py
```

You should see:
```
Starting Tomato Leaf Disease Detection Server...
Open your browser to: http://localhost:5000
 * Running on http://localhost:5000
```

### Step 6: Open in Browser
Go to: **`http://localhost:5000`**

---

## 🎯 Using the Application

1. **Upload Image**
   - Click "Choose Image" button or drag & drop
   - Supported formats: JPG, JPEG, PNG
   - Maximum size: 8MB

2. **View Prediction**
   - Disease name
   - Confidence score (0-1)
   - Model accuracy on test set
   - Class-wise similarity scores (visual bar chart)

3. **Analyze Another Image**
   - Click "Analyze Another Image" to reset and upload a new image

---

## 🛑 To Stop the Server

In the terminal, press: **`Ctrl + C`**

---

## ⚙️ Technical Details

### Backend (server.py)
- **Framework**: Flask
- **API Port**: 5000
- **CORS**: Enabled (Cross-Origin Resource Sharing)
- **Endpoint**: `POST /api/predict` - accepts image file, returns prediction

### Frontend (index.html, style.css, script.js)
- **Pure HTML/CSS/JavaScript** - no dependencies
- **Modern UI** with gradient design
- **Responsive** - works on desktop and mobile
- **Drag & Drop** support for images
- **Real-time visualization** of prediction scores

### Model
- **Type**: Quantum SVM (QSVM)
- **Features**: GLCM texture + HSV color histograms + edge detection
- **Classes**: 7 tomato disease categories + healthy leaves
- **Accuracy**: ~88% on test set

---

## 📝 Supported Disease Classes

1. Tomato Early Blight
2. Tomato Late Blight
3. Tomato Leaf Mold
4. Tomato Septoria Leaf Spot
5. Tomato Bacterial Spot
6. Tomato Yellow Leaf Curl Virus
7. Healthy Leaf

---

## 🐛 Troubleshooting

**Port 5000 already in use?**
Edit `server.py` line: `app.run(debug=True, host='0.0.0.0', port=5000)`
Change `5000` to `5001` or any free port.

**Module not found errors?**
```powershell
pip install -r requirements.txt
```

**Server not connecting?**
- Ensure `server.py` is running
- Check that port 5000 is accessible
- Try opening `http://127.0.0.1:5000` instead

---

## 📂 File Overview

| File | Purpose |
|------|---------|
| `server.py` | Flask backend with API endpoints |
| `static/index.html` | Main UI page |
| `static/style.css` | Styling & layout |
| `static/script.js` | Image upload & API communication |
| `src/infer.py` | Prediction logic |
| `artifacts/model.joblib` | Trained QSVM model |

---

## 🔄 Workflow

1. User selects image in browser
2. JavaScript sends image to Flask server via `/api/predict`
3. Flask backend:
   - Validates image
   - Loads trained model
   - Extracts features
   - Runs QSVM prediction
   - Returns disease name + confidence
4. JavaScript displays results with visualization

---

Enjoy! 🍅
