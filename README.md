# 👁️ AdaptiveScreen AI v2.0

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-SocketIO-000000?style=for-the-badge&logo=flask&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-00A6FF?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**🎯 AI-powered eye tracking that adapts your screen for ultimate reading comfort**

[Features](#-features) • [Demo](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 🌟 What's New in v2.0

| Feature | Description |
|---------|-------------|
| 🔌 **WebSocket Real-time** | Instant updates using Flask-SocketIO (no polling lag) |
| 🌓 **Dark/Light Mode** | Toggle themes with `T` key |
| 📚 **Content Library** | 4 full articles across categories |
| 📊 **Analytics Dashboard** | Charts, statistics, session history |
| 👁️ **Comfort Gauge** | Animated circular comfort score |
| 🎯 **Focus Mode** | Highlight current paragraph while reading |
| ⌨️ **Keyboard Shortcuts** | Full keyboard navigation |
| ⚠️ **Eye Fatigue Alerts** | Smart notifications when strain detected |
| 🔒 **Lock/Unlock** | Lock = Camera OFF, Unlock = Adaptive tracking |

---

## ✨ Features

### 🎨 Modern UI/UX
- **Beautiful gradient design** with glassmorphism effects
- **Responsive layout** works on all screen sizes
- **Smooth animations** and transitions
- **Reading progress bar** at the top

### 👁️ Eye Tracking Technology
- **MediaPipe Face Mesh** - 478 facial landmark tracking
- **Eye openness detection** - Monitors squinting and strain
- **Real-time adaptation** - Font size changes based on eye comfort
- **Per-user calibration** - Personalized eye tracking profiles

### 📖 Smart Reader
- **Adaptive font sizing** - Text grows when you strain, shrinks when relaxed
- **Line height control** - Adjustable spacing
- **Brightness control** - System brightness adaptation
- **Sensitivity slider** - Control how responsive the adaptation is

### 📊 Analytics Dashboard
- **Comfort trends chart** - Weekly eye comfort visualization
- **Session history** - Track all your reading sessions
- **AI insights** - Smart recommendations for better reading habits
- **Stats overview** - Total time, articles read, comfort score

### ⌨️ Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `T` | Toggle dark/light theme |
| `F` | Toggle focus mode |
| `L` | Open content library |
| `Space` | Lock/Unlock tracking |
| `+` / `-` | Increase/decrease font |
| `Esc` | Toggle fullscreen |
| `?` | Show shortcuts |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Webcam
- Modern browser (Chrome/Edge recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/AdaptiveScreenAI.git
cd AdaptiveScreenAI

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Open in Browser
```
http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
AdaptiveScreenAI/
├── app.py                 # Main Flask-SocketIO application
├── requirements.txt       # Python dependencies
├── templates/
│   ├── index.html        # Login/Signup page
│   ├── reader.html       # Main reading interface
│   ├── calibrate.html    # Eye calibration wizard
│   └── dashboard.html    # Analytics dashboard
└── README.md
```

---

## 🎮 Usage

### 1. Login or Create Account
- Use **demo/demo** for quick testing
- Or create your own account for personalized calibration

### 2. Calibrate (First Time)
- Follow the 3-step wizard to calibrate for your eyes
- Capture "Open Eyes" and "Squinting" states

### 3. Start Reading
- Click **Adaptive** to enable eye tracking
- Font size will automatically adjust based on your comfort
- Click **Lock** to freeze settings and turn off camera

### 4. View Analytics
- Click the dashboard icon to see your reading statistics
- Track comfort trends over time

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Flask** | Web framework |
| **Flask-SocketIO** | Real-time WebSocket communication |
| **OpenCV** | Camera capture and processing |
| **MediaPipe** | Face mesh and eye tracking |
| **Chart.js** | Dashboard visualizations |
| **Font Awesome** | Icons |

---

## 🔧 Configuration

### Calibration
The system stores calibration data per user in `data.json`:
```json
{
  "users": {
    "username": {
      "calibration": {
        "open": 12.0,
        "squint": 5.0
      }
    }
  }
}
```

### Font Size Range
- Minimum: 14px
- Maximum: 48px
- Default: 24px

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Your Name**

- GitHub: [@Hruday-Kumar](https://github.com/Hruday-Kumar)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ for better eye health

</div>

