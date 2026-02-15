#  Edge-Based Real-Time Threat Detection System

An **Edge AI powered surveillance intelligence system** designed to detect aggressive behavior, weaponized intent, and suspicious loitering in real-time from video frames. The system performs on-device inference and triggers alerts based on dynamic risk scoring.

---

##  System Overview

The system processes each incoming frame through multiple specialized AI branches, analyzes behavior over time, computes a risk score, and generates alerts on the edge device.
Input Frame
├── Pose Branch → Aggression Score

├── Weapon Branch → Weapon + Intent Score

├── Identity Branch → Known vs Unknown

└── Tracking Module → Persistent ID over time

↓

Behavioral Analyzer

├── Violence detection

├── Weaponized intent detection

└── Loitering detection (unknown only)

↓

Risk Scoring Engine

↓

Edge Device Alert System

---

##  Core Modules

### 1. Input Processing
Handles live video frame ingestion and distributes the frame to multiple inference branches.

---

### 2. Pose Branch
- Detects human pose and body keypoints  
- Estimates **aggression score** based on posture, motion, and interaction patterns  

---

### 3. Weapon Branch
- Detects presence of weapons  
- Computes **weapon confidence score**  
- Estimates **intent score** based on pose + weapon context  

---

### 4. Identity Branch
- Performs face/person recognition  
- Classifies individuals as:
  - **Known**
  - **Unknown**
- Unknown identities are monitored more strictly  

---

### 5. Tracking Module
- Assigns **persistent ID** to each detected individual  
- Maintains temporal history across frames  
- Enables behavior-based threat modeling  

---

##  Behavioral Analyzer

Combines outputs from all branches and evaluates behavioral patterns.

### Violence Detection
Detects:
- Physical aggression
- Fighting patterns
- Sudden hostile motion

### Weaponized Intent Detection
Triggers when:
- Weapon detected + aggressive posture
- Suspicious weapon handling behavior

### Loitering Detection *(Unknown Only)*
Flags:
- Prolonged presence in restricted zone
- Suspicious stationary movement
- Repeated roaming pattern

---

##  Risk Scoring Engine

Aggregates:
- Aggression score
- Weapon score
- Intent score
- Identity status
- Behavioral signals

Produces a **dynamic threat level**:
- Low
- Medium
- High
- Critical

---

##  Edge Device Alert System

Triggers real-time alerts based on risk level.

**Possible Actions**
- Push notification
- Alarm trigger
- Snapshot capture
- Video clip recording
- Send alert to monitoring dashboard

---

##  Key Features

- Real-time edge inference  
- Multi-branch AI architecture  
- Persistent identity tracking  
- Behavioral intelligence over time  
- Dynamic risk scoring  
- Low-latency alerting system  
- Works without constant cloud dependency  

---

##  Future Improvements

- Multi-camera cross tracking  
- Federated learning for edge updates  
- Audio threat detection  
- Crowd anomaly detection  
- Adaptive risk calibration  

---

##  Tech Stack (Example)

- Python / C++
- OpenCV
- TensorRT / ONNX Runtime
- PyTorch / TensorFlow
- DeepSORT / ByteTrack
- Edge Device: Jetson


