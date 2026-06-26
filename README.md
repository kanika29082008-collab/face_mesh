<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&pause=1000&color=00C2FF&center=true&vCenter=true&width=700&lines=😎+Face+Mesh+System;Real-Time+Facial+Landmark+Detection;Built+with+Python+%2B+OpenCV+%2B+MediaPipe" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python\&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge\&logo=opencv\&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-orange?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?style=for-the-badge\&logo=numpy)

<br/>

> **🤖 Real-Time Face Tracking & Facial Analysis System**
>
> Detect landmarks → Analyze expressions → Estimate head pose → Apply AR filters

<br/>

![GitHub stars](https://img.shields.io/github/stars/kanika29082008-collab/Face-Mesh-System?style=social)
![GitHub forks](https://img.shields.io/github/forks/kanika29082008-collab/Face-Mesh-System?style=social)
![License](https://img.shields.io/badge/License-MIT-green.svg)

</div>

---

# ✨ What is Face Mesh System?

**Face Mesh System** is a real-time computer vision application that detects facial landmarks and performs advanced face analysis using MediaPipe and OpenCV.

The system can identify facial expressions, estimate head orientation, apply AR filters, and track facial movement with high accuracy and performance.

---

# 🚀 Features

| Feature                 | Description                       |
| ----------------------- | --------------------------------- |
| 🎯 Face Mesh Detection  | Detects 468 facial landmarks      |
| 😎 AR Filters           | Sunglasses overlay & face effects |
| 😊 Expression Analysis  | Blink, Smile & Mouth detection    |
| 🎭 Head Pose Estimation | Pitch, Roll & Yaw calculation     |
| 📊 CSV Logging          | Frame-wise metrics export         |
| 🎥 Video Recording      | Save processed sessions           |
| ⚡ High FPS              | Optimized real-time performance   |
| 🎨 Wireframe Overlay    | Full facial mesh visualization    |

---

# 🛠️ Tech Stack

```text
🐍 Python 3.10
📷 OpenCV
🤖 MediaPipe Face Mesh
🔢 NumPy
📊 Pandas
🎥 Video Processing
```

---

# 📂 Project Structure

```text
Face-Mesh-System/
│
├── main.py
├── capture.py
├── mesh.py
├── compositor.py
├── logger.py
├── config.py
├── requirements.txt
│
├── modules/
│   ├── filters.py
│   ├── landmarks.py
│   ├── expression.py
│   └── pose.py
│
├── assets/
│   ├── sunglasses.png
│   ├── facemesh-demo.png
│   ├── wireframe-demo.png
│   └── pose-demo.png
│
├── output/
│   ├── metrics.csv
│   └── recordings/
│
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/kanika29082008-collab/Face-Mesh-System.git

cd Face-Mesh-System
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Project

```bash
python main.py
```

---

# 📊 Sample Output

```json
{
  "EAR": 0.31,
  "MAR": 0.42,
  "Blink": false,
  "Smile": true,
  "HeadPose": {
    "Pitch": 5.2,
    "Yaw": -2.1,
    "Roll": 1.8
  }
}
```

---

# 📸 Screenshots

<div align="center">

|                 😎 Sunglasses Filter                |                    🎯 Face Mesh                   |
| :-------------------------------------------------: | :-----------------------------------------------: |
| <img src="assets/sunglasses-demo.png" width="300"/> | <img src="assets/facemesh-demo.png" width="300"/> |

<br/>

|                  🎭 Head Pose                 |                    🎨 Wireframe                    |
| :-------------------------------------------: | :------------------------------------------------: |
| <img src="assets/pose-demo.png" width="300"/> | <img src="assets/wireframe-demo.png" width="300"/> |

</div>

---

# 🎥 Demo Video

```text
Upload your demo video here
```

Example:

```markdown
https://github.com/user-attachments/assets/demo-video.mp4
```

---

# 📈 Performance

| Metric             | Value |
| ------------------ | ----- |
| FPS                | ~60   |
| Landmarks          | 468   |
| Real-Time          | Yes   |
| Multi-Face Support | Yes   |

---

# 🚀 Future Improvements

* [x] Face Mesh Detection
* [x] AR Filters
* [x] Blink Detection
* [x] Head Pose Estimation
* [x] Video Recording
* [ ] Emotion Recognition
* [ ] Face Authentication
* [ ] Streamlit Dashboard
* [ ] Web Deployment
* [ ] Mobile Integration

---

# 👩‍💻 Author

<div align="center">

## Kanika Verma

**B.Tech CSE (AI & Data Science)**

Passionate about:
AI • Machine Learning • Generative AI • Computer Vision

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge\&logo=linkedin\&logoColor=white)](https://www.linkedin.com/in/kanika-verma-a00527312/)

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge\&logo=github\&logoColor=white)](https://github.com/kanika29082008-collab)

</div>

---

# 📜 License

MIT License — Free to use, modify, and distribute.

---

<div align="center">

### Made with ❤️ by Kanika Verma

⭐ If you like this project, don't forget to star the repository!

</div>
