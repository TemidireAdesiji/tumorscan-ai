# 🧠 TumorScan AI

> An AI-powered web application for detecting brain tumors using MRI scans. Built with Flask and deep learning, supporting multiclass tumor classification: **Meningioma**, **Glioma**, **Pituitary**, and **None**.

---

## 🚀 Features

- 🧬 Multiclass brain tumor detection (4 types)
- 📷 Upload MRI scans and get predictions instantly
- ⚙️ Deep learning model (ResNet / CNN)
- 🌐 Flask-based web interface with Bootstrap 5
- 📁 Image preprocessing and prediction in real-time
- 📦 Model training script included (`train.py`)

---

## 🧰 Tech Stack

| Component    | Technology      |
|-------------|------------------|
| Backend     | Flask (Python)   |
| ML Framework| PyTorch or TensorFlow |
| Frontend    | HTML5 + Bootstrap 5 |
| Deployment  | Localhost |
| Model Input | MRI Brain Images (JPG/PNG) |

---

## 📂 Project Structure

```

TumorScan-AI/
│
├── app.py                # Flask web server
├── model.py              # Prediction logic (PyTorch)
├── train.py              # CNN model training (TensorFlow)
├── templates/            # HTML templates (Bootstrap 5)
│   ├── DiseaseDet.html
│   ├── uimg.html
│   ├── pred.html
│   └── error.html
├── static/
│   ├── uploads/          # Uploaded MRI images
│   └── b.jpg             # Homepage image
├── models/               # Saved models (.pt or .h5)
├── dataset/              # Training/test dataset
│   ├── train/
│   └── test/
└── README.md

````

---

## 🧪 How to Use Locally

### 1. Clone the repo

```bash
git clone https://github.com/TemidireAdesiji/tumorscan-ai.git
cd tumorscan-ai
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> Make sure `torch`, `torchvision`, `flask`, and `Pillow` are included.

### 3. Start the app

```bash
python app.py
```

Open your browser to [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 Model Training (Optional)

If you're using the included `train.py`:

```bash
python train.py
```

This script:

* Trains a CNN on 64x64 MRI images
* Saves the best model to `model/brain_tumor_model.h5`
* Expects `dataset/train/` and `dataset/test/` directories with 4 subfolders each
