"""
AdaptiveScreen AI - Clean Rebuild
Real-time eye-tracking with WebSocket communication
Single source of truth, proper camera control
"""

import cv2
import json
import hashlib
import threading
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Callable

import mediapipe as mp
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_socketio import SocketIO, emit

# Try to import screen brightness control
try:
    import screen_brightness_control as sbc
    HAS_BRIGHTNESS = True
except ImportError:
    HAS_BRIGHTNESS = False
    print("⚠️ screen_brightness_control not available - brightness control disabled")

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "data.json"

# Eye landmark indices (MediaPipe Face Mesh)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Smoothing factors
FONT_SMOOTH = 0.15
BRIGHTNESS_SMOOTH = 0.2

# ============================================================
# DATA STORAGE
# ============================================================

ANALYTICS_FILE = BASE_DIR / "analytics.json"

def load_data() -> Dict:
    """Load all data from JSON file"""
    if not DATA_FILE.exists():
        default = {
            "users": {
                "demo": {
                    "name": "Demo User",
                    "password": hashlib.sha256("demo".encode()).hexdigest(),
                    "calibration": {"open": 12.0, "squint": 5.0}
                }
            }
        }
        save_data(default)
        return default
    return json.loads(DATA_FILE.read_text())

def save_data(data: Dict):
    """Save all data to JSON file"""
    DATA_FILE.write_text(json.dumps(data, indent=2))

def load_analytics() -> Dict:
    """Load analytics data"""
    if not ANALYTICS_FILE.exists():
        default = {"sessions": [], "daily_stats": {}}
        save_analytics(default)
        return default
    return json.loads(ANALYTICS_FILE.read_text())

def save_analytics(data: Dict):
    """Save analytics data"""
    ANALYTICS_FILE.write_text(json.dumps(data, indent=2))

# ============================================================
# SESSION TRACKER - Real-time analytics
# ============================================================

class SessionTracker:
    """Track reading session analytics in real-time"""
    
    def __init__(self):
        self.active = False
        self.start_time = None
        self.article = "Unknown"
        self.font_sizes = []
        self.comfort_scores = []
        self.adjustments = 0
        self.strain_events = 0
        self.last_font = 24
    
    def start(self, article: str = "The Future of Human-Computer Interaction"):
        """Start a new session"""
        self.active = True
        self.start_time = time.time()
        self.article = article
        self.font_sizes = []
        self.comfort_scores = []
        self.adjustments = 0
        self.strain_events = 0
        self.last_font = 24
    
    def record(self, font_size: int, comfort: int, eye_openness: float):
        """Record a data point"""
        if not self.active:
            return
        
        self.font_sizes.append(font_size)
        self.comfort_scores.append(comfort)
        
        # Track adjustments
        if abs(font_size - self.last_font) > 1:
            self.adjustments += 1
        self.last_font = font_size
        
        # Track strain events (low eye openness)
        if eye_openness < 8:
            self.strain_events += 1
    
    def end(self, username: str):
        """End session and save to analytics"""
        if not self.active or not self.start_time:
            return
        
        duration = int(time.time() - self.start_time)
        if duration < 5:  # Skip very short sessions
            self.active = False
            return
        
        avg_font = sum(self.font_sizes) / len(self.font_sizes) if self.font_sizes else 24
        avg_comfort = sum(self.comfort_scores) / len(self.comfort_scores) if self.comfort_scores else 75
        
        # Determine comfort rating
        if avg_comfort >= 80:
            rating = "Excellent"
        elif avg_comfort >= 60:
            rating = "Good"
        elif avg_comfort >= 40:
            rating = "Fair"
        else:
            rating = "Poor"
        
        session_data = {
            "username": username,
            "article": self.article,
            "duration": duration,
            "avg_font": round(avg_font),
            "avg_comfort": round(avg_comfort),
            "adjustments": self.adjustments,
            "strain_events": self.strain_events,
            "rating": rating,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Save to analytics
        analytics = load_analytics()
        analytics["sessions"].insert(0, session_data)  # Most recent first
        analytics["sessions"] = analytics["sessions"][:50]  # Keep last 50 sessions
        
        # Update daily stats
        today = time.strftime("%Y-%m-%d")
        if today not in analytics["daily_stats"]:
            analytics["daily_stats"][today] = {
                "total_time": 0,
                "articles_read": 0,
                "avg_comfort": 0,
                "strain_events": 0,
                "comfort_samples": []
            }
        
        daily = analytics["daily_stats"][today]
        daily["total_time"] += duration
        daily["articles_read"] += 1
        daily["strain_events"] += self.strain_events
        daily["comfort_samples"].append(avg_comfort)
        daily["avg_comfort"] = round(sum(daily["comfort_samples"]) / len(daily["comfort_samples"]))
        
        save_analytics(analytics)
        self.active = False
        print(f"📊 Session saved: {duration}s, comfort={avg_comfort:.0f}%, adjustments={self.adjustments}")

# Global session tracker
session_tracker = SessionTracker()

# ============================================================
# EYE TRACKER - CLEAN IMPLEMENTATION
# ============================================================

@dataclass
class DisplayState:
    """Current display state - single source of truth"""
    font_size: int = 24
    brightness: int = 60
    comfort: int = 75
    eye_openness: float = 0.0
    status: str = "Initializing..."
    is_tracking: bool = False
    is_locked: bool = False
    face_detected: bool = False
    camera_on: bool = False
    tracking: bool = False

class EyeTracker:
    """Clean eye tracker with proper start/stop"""
    
    def __init__(self):
        self.camera: Optional[cv2.VideoCapture] = None
        self.face_mesh = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        # Calibration values (will be set per user)
        self.open_value = 12.0
        self.squint_value = 5.0
        
        # Smoothed values
        self._smoothed_font = 24.0
        self._smoothed_brightness = 60.0
        
        # Current state
        self.state = DisplayState()
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # Callback for state updates
        self.on_update: Optional[Callable[[DisplayState], None]] = None
    
    def set_calibration(self, open_val: float, squint_val: float):
        """Set calibration values for current user"""
        with self._lock:
            self.open_value = max(open_val, squint_val + 1)
            self.squint_value = squint_val
            print(f"📐 Calibration set: open={self.open_value:.2f}, squint={self.squint_value:.2f}")
    
    def start(self) -> bool:
        """Start the camera and tracking"""
        # Already running? Just return success
        if self.running:
            print("📷 Camera already running")
            return True
        
        # Use lock to prevent race conditions
        with self._lock:
            if self.running:
                return True
            
            print("📷 Starting camera...")
            
            # Release any existing camera first
            if self.camera is not None:
                try:
                    self.camera.release()
                except:
                    pass
                self.camera = None
            
            # Try opening camera with retries
            for attempt in range(3):
                try:
                    self.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    if self.camera.isOpened():
                        break
                    print(f"📷 Camera attempt {attempt + 1} failed, retrying...")
                    time.sleep(0.5)
                except Exception as e:
                    print(f"📷 Camera error: {e}")
                    time.sleep(0.5)
            
            if not self.camera or not self.camera.isOpened():
                print("❌ Failed to open camera after retries")
                self.state.status = "Camera not available"
                self.state.camera_on = False
                return False
            
            # Initialize face mesh
            try:
                self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            except Exception as e:
                print(f"❌ Failed to init face mesh: {e}")
                self.camera.release()
                self.camera = None
                return False
            
            self.running = True
            self.state.is_tracking = True
            self.state.is_locked = False
            self.state.camera_on = True
            self.state.tracking = True
            self.state.status = "Tracking active"
        
        # Start processing thread outside lock
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        
        print("✅ Camera started, tracking active")
        return True
    
    def stop(self):
        """Stop the camera completely"""
        if not self.running:
            return
        
        print("📷 Stopping camera...")
        
        self.running = False
        
        # Wait for thread to finish
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Close face mesh
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
        
        self.state.is_tracking = False
        self.state.camera_on = False
        self.state.tracking = False
        self.state.status = "Camera stopped"
        self.state.face_detected = False
        
        print("✅ Camera stopped")
    
    def lock(self):
        """Lock current values and stop camera"""
        with self._lock:
            self.state.is_locked = True
        self.stop()
        self.state.status = f"LOCKED: {self.state.font_size}px | {self.state.brightness}%"
        print(f"🔒 Locked at font={self.state.font_size}px, brightness={self.state.brightness}%")
    
    def unlock(self):
        """Unlock and resume tracking"""
        with self._lock:
            self.state.is_locked = False
        # Only start if not already running
        if not self.running:
            self.start()
        else:
            self.state.status = "Tracking active"
        print("🔓 Unlocked, tracking resumed")
    
    def _process_loop(self):
        """Main processing loop - runs in separate thread"""
        openness_history = []
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process face
            results = self.face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                
                # Calculate eye openness
                openness = self._calculate_openness(landmarks, h)
                openness_history.append(openness)
                if len(openness_history) > 8:
                    openness_history.pop(0)
                
                avg_openness = sum(openness_history) / len(openness_history)
                
                # Map to font size and brightness
                font_size = self._map_value(avg_openness, self.squint_value, self.open_value, 48, 18)
                brightness = self._map_value(avg_openness, self.squint_value, self.open_value, 100, 30)
                comfort = self._map_value(avg_openness, self.squint_value, self.open_value, 40, 100)
                
                # Smooth the values
                self._smoothed_font = self._smoothed_font * (1 - FONT_SMOOTH) + font_size * FONT_SMOOTH
                self._smoothed_brightness = self._smoothed_brightness * (1 - BRIGHTNESS_SMOOTH) + brightness * BRIGHTNESS_SMOOTH
                
                # Update state
                with self._lock:
                    self.state.font_size = int(self._smoothed_font)
                    self.state.brightness = int(self._smoothed_brightness)
                    self.state.comfort = int(comfort)
                    self.state.eye_openness = avg_openness
                    self.state.face_detected = True
                    self.state.status = "Tracking active"
                
                # Record data for analytics
                session_tracker.record(
                    font_size=int(self._smoothed_font),
                    comfort=int(comfort),
                    eye_openness=avg_openness
                )
                
                # Apply system brightness
                if HAS_BRIGHTNESS:
                    try:
                        sbc.set_brightness(int(self._smoothed_brightness))
                    except:
                        pass
            else:
                with self._lock:
                    self.state.face_detected = False
                    self.state.status = "Face not detected"
                    self.state.eye_openness = 0
            
            # Notify callback
            if self.on_update:
                self.on_update(self.state)
            
            time.sleep(0.033)  # ~30 FPS
    
    def _calculate_openness(self, landmarks, height: int) -> float:
        """Calculate eye openness from landmarks"""
        # Left eye
        left_top = landmarks[LEFT_EYE_TOP].y * height
        left_bottom = landmarks[LEFT_EYE_BOTTOM].y * height
        left_open = abs(left_bottom - left_top)
        
        # Right eye
        right_top = landmarks[RIGHT_EYE_TOP].y * height
        right_bottom = landmarks[RIGHT_EYE_BOTTOM].y * height
        right_open = abs(right_bottom - right_top)
        
        return (left_open + right_open) / 2
    
    def _map_value(self, value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
        """Map value from one range to another"""
        if in_max == in_min:
            return out_min
        ratio = (value - in_min) / (in_max - in_min)
        ratio = max(0, min(1, ratio))
        return out_min + ratio * (out_max - out_min)
    
    def get_state(self) -> DisplayState:
        """Get current state (thread-safe)"""
        with self._lock:
            return DisplayState(
                font_size=self.state.font_size,
                brightness=self.state.brightness,
                comfort=self.state.comfort,
                eye_openness=self.state.eye_openness,
                status=self.state.status,
                is_tracking=self.state.is_tracking,
                is_locked=self.state.is_locked,
                face_detected=self.state.face_detected
            )

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)
app.secret_key = 'adaptive-screen-ai-2025-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global tracker instance
tracker = EyeTracker()

def broadcast_state():
    """Broadcast current state to all clients"""
    state = tracker.get_state()
    socketio.emit('state_update', asdict(state))

# Set tracker callback
tracker.on_update = lambda s: socketio.emit('state_update', asdict(s))

# ============================================================
# ROUTES - PAGES
# ============================================================

@app.route('/')
def index():
    """Home/Login page"""
    if 'user' in session:
        return redirect(url_for('reader'))
    return render_template('index.html')

@app.route('/reader')
def reader():
    """Main reader page"""
    return render_template('reader.html', user=session.get('user'))

@app.route('/calibrate')
def calibrate():
    """Calibration page"""
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('calibrate.html', user=session.get('user'))

@app.route('/dashboard')
def dashboard():
    """Analytics dashboard page"""
    return render_template('dashboard.html', user=session.get('user'))

# ============================================================
# ROUTES - API
# ============================================================

@app.route('/api/login', methods=['POST'])
def login():
    """Login user"""
    data = request.get_json()
    username = data.get('username', '').lower().strip()
    password = data.get('password', '')
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'})
    
    db = load_data()
    user = db['users'].get(username)
    
    if not user:
        return jsonify({'success': False, 'error': 'User not found'})
    
    if user['password'] != hashlib.sha256(password.encode()).hexdigest():
        return jsonify({'success': False, 'error': 'Invalid password'})
    
    session['user'] = username
    
    # Load user's calibration
    if 'calibration' in user:
        tracker.set_calibration(user['calibration']['open'], user['calibration']['squint'])
    
    return jsonify({'success': True, 'redirect': '/reader'})

@app.route('/api/signup', methods=['POST'])
def signup():
    """Create new user"""
    data = request.get_json()
    username = data.get('username', '').lower().strip()
    password = data.get('password', '')
    name = data.get('name', username)
    
    if not username or not password:
        return jsonify({'success': False, 'error': 'Username and password required'})
    
    if len(username) < 3:
        return jsonify({'success': False, 'error': 'Username must be at least 3 characters'})
    
    db = load_data()
    
    if username in db['users']:
        return jsonify({'success': False, 'error': 'Username already exists'})
    
    db['users'][username] = {
        'name': name,
        'password': hashlib.sha256(password.encode()).hexdigest(),
        'calibration': {'open': 12.0, 'squint': 5.0}
    }
    save_data(db)
    
    session['user'] = username
    return jsonify({'success': True, 'redirect': '/calibrate'})

@app.route('/api/logout', methods=['POST'])
def logout():
    """Logout user"""
    username = session.get('user', 'anonymous')
    # Save any active session before logout
    session_tracker.end(username)
    session.pop('user', None)
    tracker.stop()
    return jsonify({'success': True})

@app.route('/api/calibration', methods=['POST'])
def save_calibration():
    """Save user's calibration"""
    if 'user' not in session:
        return jsonify({'success': False, 'error': 'Not logged in'})
    
    data = request.get_json()
    open_val = float(data.get('open', 12.0))
    squint_val = float(data.get('squint', 5.0))
    
    db = load_data()
    username = session['user']
    
    if username in db['users']:
        db['users'][username]['calibration'] = {'open': open_val, 'squint': squint_val}
        save_data(db)
        tracker.set_calibration(open_val, squint_val)
    
    return jsonify({'success': True})

@app.route('/api/state')
def get_state():
    """Get current state (for initial load)"""
    return jsonify(asdict(tracker.get_state()))

@app.route('/api/analytics')
def get_analytics():
    """Get analytics data for dashboard"""
    username = session.get('user')
    analytics = load_analytics()
    
    # Filter sessions for current user (or all if no user)
    if username:
        user_sessions = [s for s in analytics.get('sessions', []) if s.get('user') == username]
    else:
        user_sessions = analytics.get('sessions', [])
    
    # Calculate summary stats
    total_time = sum(s.get('duration', 0) for s in user_sessions)
    total_sessions = len(user_sessions)
    avg_comfort = 0
    total_adjustments = sum(s.get('adjustments', 0) for s in user_sessions)
    total_strain = sum(s.get('strain_events', 0) for s in user_sessions)
    
    if user_sessions:
        comfort_scores = [s.get('avg_comfort', 75) for s in user_sessions if s.get('avg_comfort')]
        if comfort_scores:
            avg_comfort = round(sum(comfort_scores) / len(comfort_scores), 1)
    
    # Get last 7 sessions for charts
    recent = user_sessions[-7:] if len(user_sessions) >= 7 else user_sessions
    
    # Prepare chart data
    labels = []
    durations = []
    comforts = []
    font_sizes = []
    
    for s in recent:
        # Format date label
        ts = s.get('timestamp', '')
        if ts:
            try:
                dt = datetime.fromisoformat(ts)
                labels.append(dt.strftime('%b %d'))
            except:
                labels.append('Session')
        else:
            labels.append('Session')
        
        durations.append(round(s.get('duration', 0) / 60, 1))  # Convert to minutes
        comforts.append(s.get('avg_comfort', 75))
        font_sizes.append(s.get('avg_font_size', 24))
    
    return jsonify({
        'summary': {
            'total_time': round(total_time / 60, 1),  # Minutes
            'total_sessions': total_sessions,
            'avg_comfort': avg_comfort,
            'total_adjustments': total_adjustments,
            'total_strain': total_strain
        },
        'charts': {
            'labels': labels,
            'durations': durations,
            'comforts': comforts,
            'font_sizes': font_sizes
        },
        'recent_sessions': recent[-5:]  # Last 5 for table
    })

# ============================================================
# WEBSOCKET EVENTS
# ============================================================

@socketio.on('connect')
def on_connect():
    """Client connected"""
    print(f"🔌 Client connected")
    emit('state_update', asdict(tracker.get_state()))

@socketio.on('start_tracking')
def on_start_tracking(data=None):
    """Start eye tracking"""
    print("▶️ Start tracking requested")
    article = "Unknown Article"
    if data and isinstance(data, dict):
        article = data.get('article', 'Unknown Article')
    session_tracker.start(article)
    if tracker.start():
        emit('state_update', asdict(tracker.get_state()))

@socketio.on('stop_tracking')
def on_stop_tracking():
    """Stop eye tracking"""
    print("⏹️ Stop tracking requested")
    tracker.stop()
    # End session and save analytics
    username = session.get('user', 'anonymous')
    session_tracker.end(username)
    emit('state_update', asdict(tracker.get_state()))

@socketio.on('lock')
def on_lock():
    """Lock current settings"""
    print("🔒 Lock requested")
    tracker.lock()
    emit('state_update', asdict(tracker.get_state()))

@socketio.on('unlock')
def on_unlock():
    """Unlock and resume tracking"""
    print("🔓 Unlock requested")
    tracker.unlock()
    emit('state_update', asdict(tracker.get_state()))

@socketio.on('set_font')
def on_set_font(data):
    """Manually set font size"""
    size = int(data.get('size', 24))
    tracker.state.font_size = size
    tracker._smoothed_font = float(size)
    emit('state_update', asdict(tracker.get_state()))

# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("""
╔══════════════════════════════════════════════════════════════╗
║           AdaptiveScreen AI - Clean Rebuild                  ║
╠══════════════════════════════════════════════════════════════╣
║  🌐 Open in browser: http://127.0.0.1:5000                   ║
║  📖 Reader: http://127.0.0.1:5000/reader                     ║
║                                                              ║
║  Features:                                                   ║
║  • Real-time WebSocket updates (no polling)                  ║
║  • Lock = Camera OFF, values frozen                          ║
║  • Unlock = Camera ON, adaptive tracking                     ║
║  • Per-user calibration                                      ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    socketio.run(app, host='127.0.0.1', port=5000, debug=True, use_reloader=False)
